"""The deployable student: one CNN forward pass -> crop gate + per-crop open-set disease heads.

    image ─▶ CNN backbone ─▶ feature map F (C×h×w) ─▶ GAP ─▶ embedding e (C)
                                   │                         ├─▶ gate head (linear): rice|tea|chili|other_plant|non_plant
                                   │                         └─▶ disease head per crop (cosine classifier, AM-Softmax in training)
                                   └─▶ CAM for any class = Σ_k W[c,k]·F_k  (exact: heads are linear in GAP(F))

Design choices (docs/ARCHITECTURE.md, docs/RESEARCH.md):
  * cosine heads + additive margin keep each known class compact, leaving room in feature space to
    flag unknown diseases (Meng et al. 2023);
  * the embedding is used for the prototype library (Mahalanobis / kNN), so an officer-verified new
    disease can be added without retraining;
  * pooling is plain GAP over the backbone's last feature map so CAM is exact and gradient-free,
    which keeps the ONNX/INT8 serving path simple.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

STUDENT_BACKBONES = {
    "mnv4_s": "mobilenetv4_conv_small.e2400_r224_in1k",
    "mnv4_m": "mobilenetv4_conv_medium.e500_r224_in1k",
    "effnet_b0": "efficientnet_b0.ra_in1k",
    "convnext_atto": "convnext_atto.d2_in1k",
}


class CosineHead(nn.Module):
    """Logits = scale · cos(e, w_c). With ``margin`` > 0 and labels given: AM-Softmax logits."""

    def __init__(self, dim: int, n_classes: int, scale: float = 30.0, margin: float = 0.2) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, dim) * 0.01)
        self.scale, self.margin = scale, margin

    def cosine(self, e: torch.Tensor) -> torch.Tensor:
        return F.linear(F.normalize(e, dim=1), F.normalize(self.weight, dim=1))

    def forward(self, e: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cos = self.cosine(e)
        if labels is not None and self.margin > 0:
            cos = cos - self.margin * F.one_hot(labels.clamp(min=0), cos.shape[1]).to(cos.dtype)
        return self.scale * cos


class Student(nn.Module):
    def __init__(self, backbone: str, crop_classes: dict[str, list[str]], gate_classes: list[str],
                 pretrained: bool = True, scale: float = 30.0, margin: float = 0.2, drop: float = 0.2) -> None:
        super().__init__()
        import timm

        timm_id = STUDENT_BACKBONES.get(backbone, backbone)
        self.backbone = timm.create_model(timm_id, pretrained=pretrained, num_classes=0)
        dim = self.backbone.num_features
        self.crops = list(crop_classes)
        self.crop_classes = crop_classes
        self.gate_classes = gate_classes
        self.dropout = nn.Dropout(drop)
        self.gate = nn.Linear(dim, len(gate_classes))
        self.heads = nn.ModuleDict({c: CosineHead(dim, len(k), scale, margin) for c, k in crop_classes.items()})
        self.embed_dim = dim
        self.timm_id = timm_id

    def features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fmap = self.backbone.forward_features(x)
        if fmap.ndim != 4:
            raise ValueError("student backbones must return an NCHW feature map (use a CNN)")
        return fmap, fmap.mean((2, 3))

    def forward(self, x: torch.Tensor, disease_labels: dict[str, torch.Tensor] | None = None) -> dict:
        fmap, emb = self.features(x)
        e = self.dropout(emb)
        out = {"fmap": fmap, "emb": emb, "gate": self.gate(e), "disease": {}}
        for c, head in self.heads.items():
            out["disease"][c] = head(e, disease_labels.get(c) if disease_labels else None)
        return out

    @torch.no_grad()
    def cam(self, fmap: torch.Tensor, crop: str, class_idx: torch.Tensor) -> torch.Tensor:
        """Class activation maps (N×h×w, min-max normalised to [0, 1]) for a disease class."""
        w = F.normalize(self.heads[crop].weight, dim=1)[class_idx]          # N×C
        cam = torch.einsum("nc,nchw->nhw", w, fmap).relu()
        flat = cam.flatten(1)
        lo, hi = flat.min(1).values[:, None, None], flat.max(1).values[:, None, None]
        return (cam - lo) / (hi - lo + 1e-6)
