"""Training objective of the student.

    L = CE_gate (label smoothing)                       — crop gate on every trainable image
      + Σ_crop AM-Softmax(disease head)                 — known diseases of that crop
      + λ_oe · Σ_crop OE(disease head)                  — outlier exposure: images that are NOT
                                                          this crop are pushed to a uniform output
      + λ_kd · T² · [KL(teacher_gate‖student_gate) + KL(teacher_disease‖student_disease)]

Outlier exposure (Hendrycks et al. 2019) uses only *other* crops, other plants and non-plant
images — never the held-out ``unknown_eval`` diseases, which stay unseen for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    gate: float = 1.0
    disease: float = 1.0
    oe: float = 0.5
    kd: float = 1.0
    kd_temperature: float = 2.0
    label_smoothing: float = 0.1


def kd_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, t: float) -> torch.Tensor:
    """Hinton KD: T² · KL(softmax(teacher/T) ‖ softmax(student/T)), mean over rows."""
    if student_logits.numel() == 0:
        return student_logits.sum() * 0
    p_t = F.softmax(teacher_logits / t, dim=1)
    log_p_s = F.log_softmax(student_logits / t, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * t * t


def outlier_exposure(logits: torch.Tensor) -> torch.Tensor:
    """Cross-entropy to the uniform distribution."""
    if logits.numel() == 0:
        return logits.sum() * 0
    return -F.log_softmax(logits, dim=1).mean(1).mean()


def student_loss(model, out: dict, batch: dict, w: LossWeights) -> tuple[torch.Tensor, dict]:
    """``batch``: gate (N), crop_idx (N, -1 if not a crop image), disease (N, -1 if not a known disease),
    optional t_gate (N×G) and t_disease (N×Kmax, padded with -1e4) teacher logits."""
    logs: dict[str, float] = {}
    loss = w.gate * F.cross_entropy(out["gate"], batch["gate"], label_smoothing=w.label_smoothing)
    logs["gate"] = float(loss)

    have_teacher = "t_gate" in batch
    if have_teacher and w.kd > 0:
        kd = kd_kl(out["gate"], batch["t_gate"], w.kd_temperature)
        loss = loss + w.kd * kd
        logs["kd_gate"] = float(kd)

    emb = model.dropout(out["emb"])
    for ci, crop in enumerate(model.crops):
        head = model.heads[crop]
        own = (batch["crop_idx"] == ci) & (batch["disease"] >= 0)
        if own.any():
            am_logits = head(emb[own], batch["disease"][own])
            ce = F.cross_entropy(am_logits, batch["disease"][own])
            loss = loss + w.disease * ce
            logs[f"am_{crop}"] = float(ce)
            if have_teacher and w.kd > 0:
                k = am_logits.shape[1]
                kd = kd_kl(head(emb[own]), batch["t_disease"][own][:, :k], w.kd_temperature)
                loss = loss + w.kd * kd
                logs[f"kd_{crop}"] = float(kd)
        other = batch["crop_idx"] != ci
        if w.oe > 0 and other.any():
            oe = outlier_exposure(head(emb[other]))
            loss = loss + w.oe * oe
            logs[f"oe_{crop}"] = float(oe)
    logs["total"] = float(loss)
    return loss, logs
