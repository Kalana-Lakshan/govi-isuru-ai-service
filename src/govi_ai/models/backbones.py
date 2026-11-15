"""Frozen feature extractors used as teachers and as baselines.

Self-supervised foundation models (DINOv2) and a vision-language model (CLIP) are the teacher
candidates; frozen supervised ImageNet CNNs are included as the conventional baseline that the
literature shows collapses under lab-to-field shift (see docs/RESEARCH.md [1], [2], [4]).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackboneSpec:
    key: str
    timm_id: str
    family: str          # ssl | vlm | supervised
    params_m: float      # approx. parameters (millions), for the paper's cost table
    note: str = ""


BACKBONES: dict[str, BackboneSpec] = {
    s.key: s for s in [
        BackboneSpec("dinov2_s", "vit_small_patch14_dinov2.lvd142m", "ssl", 22, "DINOv2 ViT-S/14"),
        BackboneSpec("dinov2_b", "vit_base_patch14_dinov2.lvd142m", "ssl", 86, "DINOv2 ViT-B/14"),
        BackboneSpec("dinov2_l", "vit_large_patch14_dinov2.lvd142m", "ssl", 304, "DINOv2 ViT-L/14"),
        BackboneSpec("clip_b16", "vit_base_patch16_clip_224.openai", "vlm", 86, "CLIP ViT-B/16 image tower"),
        BackboneSpec("convnext_b", "convnext_base.fb_in22k_ft_in1k", "supervised", 89, "ImageNet-22k supervised"),
        BackboneSpec("effnet_b0", "efficientnet_b0.ra_in1k", "supervised", 5.3, "ImageNet-1k supervised"),
        BackboneSpec("mnv4_s", "mobilenetv4_conv_small.e2400_r224_in1k", "supervised", 3.8,
                     "student-sized baseline"),
    ]
}


def create_backbone(key_or_timm_id: str, img_size: int = 224, pretrained: bool = True):
    """Return ``(model, mean, std)`` with the classifier removed (``model(x)`` -> pooled features)."""
    import timm

    timm_id = BACKBONES[key_or_timm_id].timm_id if key_or_timm_id in BACKBONES else key_or_timm_id
    kwargs = {"num_classes": 0}
    if timm_id.startswith("vit_"):
        kwargs["img_size"] = img_size  # positional embeddings are interpolated by timm
    model = timm.create_model(timm_id, pretrained=pretrained, **kwargs)
    model.eval().requires_grad_(False)
    cfg = timm.data.resolve_model_data_config(model)
    return model, tuple(cfg["mean"]), tuple(cfg["std"])
