"""Fit the decision bundle from a trained student's cached outputs (phase 4).

Everything is fitted on the protocol's *validation* rows (thresholds) and *train* rows (prototypes);
test rows are only used to report what the chosen thresholds do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from govi_ai.eval import metrics as M
from govi_ai.inference.decision import CropCalibration, DecisionBundle, novelty_value
from govi_ai.openset import Mahalanobis, fit_temperature, softmax


def _low_conf_threshold(conf: np.ndarray, correct: np.ndarray, target_risk: float) -> float:
    """Smallest confidence threshold whose accepted set has error <= target_risk (max coverage)."""
    order = np.argsort(-conf)
    err = np.cumsum(1 - correct[order]) / np.arange(1, len(conf) + 1)
    ok = np.flatnonzero(err <= target_risk)
    if not ok.size:
        return float(conf.max())
    return float(conf[order][ok.max()])


NOVELTY_SCORES = ("mahalanobis", "msp", "fused")


def _loco_scores(dist: np.ndarray, probs: np.ndarray, held: int) -> tuple[np.ndarray, np.ndarray]:
    """Scores with class `held` removed from the model: its rows then play the part of an unseen disease."""
    keep = np.ones(dist.shape[1], bool)
    keep[held] = False
    maha = -dist[:, keep].min(1)
    p = probs[:, keep]
    msp = (p / p.sum(1, keepdims=True)).max(1)
    return maha, msp


def choose_novelty(dist: np.ndarray, probs: np.ndarray, y: np.ndarray, known_tpr: float,
                   maha_threshold: float, min_class_rows: int = 10) -> tuple[dict, dict]:
    """Pick the unknown-disease score for one crop without touching any unknown or test data.

    Leave-one-class-out on validation rows: each known class in turn is hidden from the prototypes and
    the softmax; its rows act as unknown and the remaining rows as known. The score with the best mean
    AUROC wins. Its threshold keeps `known_tpr` of the validation rows (full model) as known.
    """
    full_maha, full_msp = -dist.min(1), probs.max(1)
    norm = {"maha_mu": float(full_maha.mean()), "maha_sd": float(full_maha.std() + 1e-9),
            "msp_mu": float(full_msp.mean()), "msp_sd": float(full_msp.std() + 1e-9)}
    aurocs = {k: [] for k in NOVELTY_SCORES}
    for c in np.unique(y):
        held = y == c
        if held.sum() < min_class_rows or (~held).sum() < min_class_rows:
            continue
        maha, msp = _loco_scores(dist, probs, int(c))
        for kind in NOVELTY_SCORES:
            sc, _ = novelty_value({"score": kind, "threshold": 0.0, **norm}, maha, msp, maha_threshold)
            aurocs[kind].append(M.auroc(sc[~held], sc[held]))
    loco = {k: (float(np.mean(v)) if v else None) for k, v in aurocs.items()}
    scored = [k for k in NOVELTY_SCORES if loco[k] is not None]
    best = max(scored, key=lambda k: loco[k]) if scored else "mahalanobis"
    cfg = {"score": best, **norm}
    if best != "mahalanobis":
        sc, _ = novelty_value({**cfg, "threshold": 0.0}, full_maha, full_msp, maha_threshold)
        cfg["threshold"] = float(np.quantile(sc, 1 - known_tpr))
    return cfg, loco


def coverage_table(conf: np.ndarray, correct: np.ndarray,
                   coverages=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)) -> list[dict]:
    """Confidence threshold and error rate when answering a given share of photos (operating points)."""
    order = np.argsort(-conf)
    out = []
    for cov in coverages:
        n = max(1, int(round(cov * len(conf))))
        out.append({"coverage": cov, "threshold": float(conf[order][n - 1]),
                    "risk": float(1 - correct[order][:n].mean())})
    return out


def fit_bundle(df: pd.DataFrame, outs: dict, gate_classes: list[str], crop_classes: dict[str, list[str]],
               protocol: str = "split", head_weights: dict[str, np.ndarray] | None = None,
               known_tpr: float = 0.95, target_risk: float = 0.05, gate_keep: float = 0.99,
               version: str = "dev", choose_score: bool = True) -> tuple[DecisionBundle, dict]:
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
        dist_va = np.array([cal.distances(e) for e in outs["emb"][va_c]])
        cal.unknown_threshold = float(np.quantile(-dist_va.min(1), 1 - known_tpr))
        p = softmax(logits[va_c], t)
        cal.low_conf_threshold = _low_conf_threshold(p.max(1), p.argmax(1) == y[va_c], target_risk)
        loco = None
        if choose_score:
            cal.novelty, loco = choose_novelty(dist_va, p, y[va_c], known_tpr, cal.unknown_threshold)
        crops[crop] = cal

        # What these thresholds do on held-out data (for the paper / sanity).
        te_c = own & te
        unk = (df["crop"] == crop).to_numpy() & (df["role"] == "unknown_eval").to_numpy()
        r = {"temperature": t, "unknown_threshold": cal.unknown_threshold, "low_conf_threshold": cal.low_conf_threshold,
             "novelty_score": cal.novelty.get("score"), "novelty_loco_auroc_val": loco,
             "coverage_options_val": coverage_table(p.max(1), (p.argmax(1) == y[va_c]).astype(float))}

        def novelty(rows):
            d = np.array([cal.distances(e) for e in outs["emb"][rows]])
            pr = softmax(logits[rows], t)
            return novelty_value(cal.novelty, -d.min(1), pr.max(1), cal.unknown_threshold)[0], pr

        thr = novelty_value(cal.novelty, 0.0, 0.0, cal.unknown_threshold)[1]
        if te_c.any():
            s_te, p_te = novelty(te_c)
            accept = (s_te >= thr) & (p_te.max(1) >= cal.low_conf_threshold)
            corr = p_te.argmax(1) == y[te_c]
            r["test_known_flagged_unknown"] = float(np.mean(s_te < thr))
            r["test_coverage"] = float(accept.mean())
            r["test_risk_on_accepted"] = float(1 - corr[accept].mean()) if accept.any() else None
            r["test_macro_f1"] = M.macro_f1(y[te_c], p_te.argmax(1), len(classes))
            r["coverage_options_test"] = coverage_table(p_te.max(1), corr.astype(float))
        if unk.any():
            s_u, _ = novelty(unk)
            r["unknown_disease_detected"] = float(np.mean(s_u < thr))
            if te_c.any():  # reported only; never used for any choice above
                r["test_unknown_auroc"] = M.auroc(s_te, s_u)
        report[crop] = r

    meta = {"protocol": protocol, "known_tpr": known_tpr, "target_risk": target_risk, "gate_keep": gate_keep,
            "novelty_selection": "leave-one-known-class-out AUROC on validation" if choose_score else "mahalanobis"}
    return DecisionBundle(list(gate_classes), gate_t, gate_min_conf, crops, version, meta), report
