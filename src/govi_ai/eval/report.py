"""Evaluation report from model outputs (student or any model producing logits + embeddings).

Mirrors ``govi_ai.probe`` so teacher-probe and student numbers are directly comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from govi_ai.eval import metrics as M
from govi_ai.openset import Mahalanobis, energy, fit_temperature, knn_score, max_logit, msp, softmax


def crop_report(df: pd.DataFrame, emb: np.ndarray, logits: np.ndarray, classes: list[str], crop: str,
                protocol: str) -> dict:
    """``logits``: the crop's disease-head logits for every row of ``df`` (row-aligned with ``emb``)."""
    cidx = {c: i for i, c in enumerate(classes)}
    y = df["label"].map(cidx).fillna(-1).astype(int).to_numpy()
    known = ((df["crop"] == crop) & (df["role"] == "known")).to_numpy() & (y >= 0)
    tr = known & (df[protocol] == "train").to_numpy()
    va = known & (df[protocol] == "val").to_numpy()
    te = known & (df[protocol] == "test").to_numpy()
    if not te.any() or not tr.any():
        return {"skipped": f"{int(tr.sum())} train / {int(te.sum())} test rows"}

    t = fit_temperature(logits[va], y[va]) if va.any() else 1.0
    pred = logits[te].argmax(1)
    p_cal = softmax(logits[te], t)
    correct = pred == y[te]
    maha = Mahalanobis().fit(emb[tr], y[tr])

    def scores(m):
        z = logits[m]
        return {"msp": msp(z, t), "max_logit": max_logit(z), "energy": energy(z),
                "mahalanobis": maha.score(emb[m]), "knn": knn_score(emb[tr], emb[m])}

    ks = scores(te)
    in_dom_test = (df["split"] == "test").to_numpy()
    sets = {
        "unknown_disease": ((df["crop"] == crop) & (df["role"] == "unknown_eval")).to_numpy(),
        "other_crops": ((df["crop"] != crop) & (df["role"] == "known")).to_numpy() & in_dom_test,
        "other_plant": (df["role"] == "other_plant").to_numpy() & in_dom_test,
        "non_plant": (df["role"] == "non_plant").to_numpy() & in_dom_test,
    }
    open_set = {}
    for name, m in sets.items():
        if m.any():
            us = scores(m)
            open_set[name] = {"n": int(m.sum()),
                              **{k: {"auroc": M.auroc(ks[k], us[k]), "fpr95": M.fpr_at_tpr(ks[k], us[k])} for k in ks},
                              "oscr_msp": M.oscr(ks["msp"], correct, us["msp"])}
    by_source = {}
    for s in sorted(df.loc[te, "source"].unique()):
        m = (df.loc[te, "source"] == s).to_numpy()
        by_source[s] = {"n": int(m.sum()), "macro_f1": M.macro_f1(y[te][m], pred[m], len(classes))}
    return {
        "n_test": int(te.sum()), "temperature": t,
        "closed_set": {
            "accuracy": M.accuracy(y[te], pred), "macro_f1": M.macro_f1(y[te], pred, len(classes)),
            "balanced_accuracy": M.balanced_accuracy(y[te], pred, len(classes)),
            "ece_raw": M.ece(softmax(logits[te]), y[te]), "ece_calibrated": M.ece(p_cal, y[te]),
            "risk_at_80_coverage": M.risk_at_coverage(p_cal.max(1), correct, 0.8),
            "aurc": M.aurc(p_cal.max(1), correct),
            "per_class_recall": dict(zip(classes, np.nan_to_num(
                M.per_class_recall(y[te], pred, len(classes))).round(4).tolist())),
            "by_source": by_source,
        },
        "open_set": open_set,
    }


def gate_report(df: pd.DataFrame, gate_logits: np.ndarray, gate_classes: list[str], gate_idx: np.ndarray) -> dict:
    te = (df["split"] == "test").to_numpy() & (df["role"] != "unknown_eval").to_numpy()
    pred = gate_logits[te].argmax(1)
    y = gate_idx[te]
    crops_i = [i for i, c in enumerate(gate_classes) if c not in ("other_plant", "non_plant")]
    leaf = np.isin(y, crops_i)
    unk = (df["role"] == "unknown_eval").to_numpy()
    return {
        "accuracy": M.accuracy(y, pred), "macro_f1": M.macro_f1(y, pred, len(gate_classes)),
        "per_class_recall": dict(zip(gate_classes, np.nan_to_num(
            M.per_class_recall(y, pred, len(gate_classes))).round(4).tolist())),
        "confusion": M.confusion_matrix(y, pred, len(gate_classes)).tolist(),
        "wrong_crop_rate": float(np.mean(np.isin(pred[leaf], crops_i) & (pred[leaf] != y[leaf]))) if leaf.any() else None,
        "false_reject_rate": float(np.mean(~np.isin(pred[leaf], crops_i))) if leaf.any() else None,
        "unknown_disease_right_crop": float(np.mean(gate_logits[unk].argmax(1) == gate_idx[unk])) if unk.any() else None,
    }
