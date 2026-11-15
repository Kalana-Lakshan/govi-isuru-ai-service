"""Fit the decision bundle from a trained student's cached outputs (phase 4).

Everything is fitted on the protocol's *validation* rows (thresholds) and *train* rows (prototypes);
test rows are only used to report what the chosen thresholds do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from govi_ai.eval import metrics as M
from govi_ai.inference.decision import CropCalibration, DecisionBundle
from govi_ai.openset import Mahalanobis, fit_temperature, softmax


def _low_conf_threshold(conf: np.ndarray, correct: np.ndarray, target_risk: float) -> float:
    """Smallest confidence threshold whose accepted set has error <= target_risk (max coverage)."""
    order = np.argsort(-conf)
    err = np.cumsum(1 - correct[order]) / np.arange(1, len(conf) + 1)
    ok = np.flatnonzero(err <= target_risk)
    if not ok.size:
        return float(conf.max())
    return float(conf[order][ok.max()])


def fit_bundle(df: pd.DataFrame, outs: dict, gate_classes: list[str], crop_classes: dict[str, list[str]],
               protocol: str = "split", head_weights: dict[str, np.ndarray] | None = None,
               known_tpr: float = 0.95, target_risk: float = 0.05, gate_keep: float = 0.99,
               version: str = "dev") -> tuple[DecisionBundle, dict]:
    """``df`` rows are aligned with ``outs`` arrays (emb, gate, disease_<crop>) and encoded with
    ``LabelSpace.encode`` (gate_idx, disease_idx)."""
    trainable = (df["role"] != "unknown_eval").to_numpy()
    va = trainable & (df[protocol] == "val").to_numpy()
    te = trainable & (df["split"] == "test").to_numpy()
    gidx = df["gate_idx"].to_numpy()

    gate_t = fit_temperature(outs["gate"][va], gidx[va])
    p_gate = softmax(outs["gate"][va], gate_t)
    crop_rows = va & df["crop"].isin(list(crop_classes)).to_numpy()
    right = p_gate.argmax(1) == gidx[va]
    conf_right_crop = p_gate[(crop_rows[va]) & right].max(1)
    gate_min_conf = float(np.quantile(conf_right_crop, 1 - gate_keep)) if conf_right_crop.size else 0.0

    crops, report = {}, {"gate": {"temperature": gate_t, "min_conf": gate_min_conf}}
    y = df["disease_idx"].to_numpy()
    for ci, (crop, classes) in enumerate(crop_classes.items()):
        own = (df["crop"] == crop).to_numpy() & (df["role"] == "known").to_numpy() & (y >= 0)
        tr_c = own & (df[protocol] == "train").to_numpy()
        va_c = own & va
        if not tr_c.any() or not va_c.any():
            continue
        logits = outs[f"disease_{crop}"]
        t = fit_temperature(logits[va_c], y[va_c])
        maha = Mahalanobis().fit(outs["emb"][tr_c], y[tr_c])
        # Means for every class in the head order (classes without training rows get a far-away mean).
        means = np.full((len(classes), outs["emb"].shape[1]), 1e6)
        means[maha.classes_.astype(int)] = maha.means_
        cal = CropCalibration(classes=list(classes), temperature=t, means=means, precision=maha.precision_,
                              unknown_threshold=0.0, low_conf_threshold=0.0,
                              head_weights=None if head_weights is None else head_weights[crop])
        val_scores = -np.array([cal.distances(e).min() for e in outs["emb"][va_c]])
        cal.unknown_threshold = float(np.quantile(val_scores, 1 - known_tpr))
        p = softmax(logits[va_c], t)
        cal.low_conf_threshold = _low_conf_threshold(p.max(1), p.argmax(1) == y[va_c], target_risk)
        crops[crop] = cal

        # What these thresholds do on held-out data (for the paper / sanity).
        te_c = own & te
        unk = (df["crop"] == crop).to_numpy() & (df["role"] == "unknown_eval").to_numpy()
        r = {"temperature": t, "unknown_threshold": cal.unknown_threshold, "low_conf_threshold": cal.low_conf_threshold}
        if te_c.any():
            s_te = -np.array([cal.distances(e).min() for e in outs["emb"][te_c]])
            p_te = softmax(logits[te_c], t)
            accept = (s_te >= cal.unknown_threshold) & (p_te.max(1) >= cal.low_conf_threshold)
            corr = p_te.argmax(1) == y[te_c]
            r["test_known_flagged_unknown"] = float(np.mean(s_te < cal.unknown_threshold))
            r["test_coverage"] = float(accept.mean())
            r["test_risk_on_accepted"] = float(1 - corr[accept].mean()) if accept.any() else None
            r["test_macro_f1"] = M.macro_f1(y[te_c], p_te.argmax(1), len(classes))
        if unk.any():
            s_u = -np.array([cal.distances(e).min() for e in outs["emb"][unk]])
            r["unknown_disease_detected"] = float(np.mean(s_u < cal.unknown_threshold))
        report[crop] = r

    meta = {"protocol": protocol, "known_tpr": known_tpr, "target_risk": target_risk, "gate_keep": gate_keep}
    return DecisionBundle(list(gate_classes), gate_t, gate_min_conf, crops, version, meta), report
