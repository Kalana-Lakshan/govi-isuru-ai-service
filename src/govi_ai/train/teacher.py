"""Teacher soft targets for distillation, from linear heads on frozen (fused) features.

The teacher is refit *per protocol*: for a leave-one-dataset-out run, the teacher never sees the
held-out source either, otherwise its soft labels would leak test information into the student.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from govi_ai.probe import fit_head
from govi_ai.train.data import PAD, LabelSpace


def build_targets(df: pd.DataFrame, X: np.ndarray, space: LabelSpace, protocol: str, seed: int = 0) -> dict:
    """Returns {"sample_ids", "gate" (N×G), "disease" (N×Kmax, crop's own head, PAD elsewhere), "info"}.

    ``df`` must be encoded with ``LabelSpace.encode``; logits are produced for every row so the same
    file serves train and val. Logits are divided by each head's fitted temperature (calibrated
    teachers give better distillation targets).
    """
    trainable = (df["role"] != "unknown_eval").to_numpy()
    tr = trainable & (df[protocol] == "train").to_numpy()
    va = trainable & (df[protocol] == "val").to_numpy()
    info: dict = {"protocol": protocol}

    gate_head, info["gate"] = fit_head(X[tr], df["gate_idx"].to_numpy()[tr], X[va], df["gate_idx"].to_numpy()[va],
                                       space.gate_classes, seed=seed)
    gate = gate_head.logits(X) / gate_head.temperature

    disease = np.full((len(df), space.max_classes), PAD, dtype=np.float32)
    y = df["disease_idx"].to_numpy()
    for ci, crop in enumerate(space.crops):
        own = (df["crop_idx"] == ci).to_numpy() & (y >= 0)
        if not (own & tr).any():
            continue
        classes = space.crop_classes[crop]
        head, info[crop] = fit_head(X[own & tr], y[own & tr], X[own & va], y[own & va], classes, seed=seed)
        rows = (df["crop_idx"] == ci).to_numpy()
        disease[rows, :len(classes)] = head.logits(X[rows]) / head.temperature
    return {"sample_ids": df["sample_id"].to_numpy(dtype=str), "gate": gate.astype(np.float32),
            "disease": disease, "info": info}
