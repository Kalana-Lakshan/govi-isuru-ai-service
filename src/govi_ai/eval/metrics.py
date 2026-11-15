"""Metrics used throughout the paper. Pure numpy, so they run anywhere (Colab, CI, service).

Closed-set:   accuracy, macro-F1, balanced accuracy, per-class recall, confusion matrix
Calibration:  expected calibration error (ECE)
Open-set:     AUROC (known vs unknown), FPR at 95 % TPR, OSCR
Selective:    risk at a coverage level, area under the risk-coverage curve (AURC)

Convention for open-set scores: **higher score = more likely known** (e.g. max softmax prob,
max logit, negative energy).
"""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_f1(y_true, y_pred, n_classes: int) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    prec_den, rec_den = cm.sum(0), cm.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(prec_den > 0, tp / prec_den, 0.0)
        recall = np.where(rec_den > 0, tp / rec_den, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return f1


def macro_f1(y_true, y_pred, n_classes: int, present_only: bool = True) -> float:
    """Macro-F1. With ``present_only`` classes absent from ``y_true`` are ignored (needed for
    cross-dataset tests where a held-out source covers only some classes)."""
    f1 = per_class_f1(y_true, y_pred, n_classes)
    if present_only:
        present = np.bincount(np.asarray(y_true), minlength=n_classes) > 0
        f1 = f1[present]
    return float(f1.mean()) if f1.size else 0.0


def per_class_recall(y_true, y_pred, n_classes: int) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    den = cm.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, np.diag(cm) / den, np.nan)


def balanced_accuracy(y_true, y_pred, n_classes: int) -> float:
    return float(np.nanmean(per_class_recall(y_true, y_pred, n_classes)))


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def ece(probs, y_true, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins."""
    probs = np.asarray(probs, dtype=float)
    conf = probs.max(1)
    correct = probs.argmax(1) == np.asarray(y_true)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def auroc(scores_known, scores_unknown) -> float:
    """P(score_known > score_unknown) with ties counted half (Mann-Whitney U)."""
    k = np.asarray(scores_known, dtype=float)
    u = np.asarray(scores_unknown, dtype=float)
    if k.size == 0 or u.size == 0:
        return float("nan")
    allv = np.concatenate([k, u])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sorted_v = allv[order]
    # average ranks for ties
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_k = ranks[: k.size].sum()
    return float((r_k - k.size * (k.size + 1) / 2) / (k.size * u.size))


def fpr_at_tpr(scores_known, scores_unknown, tpr: float = 0.95) -> float:
    """Fraction of unknowns accepted when the threshold keeps ``tpr`` of knowns."""
    k = np.sort(np.asarray(scores_known, dtype=float))
    u = np.asarray(scores_unknown, dtype=float)
    if k.size == 0 or u.size == 0:
        return float("nan")
    thr = k[int(np.floor((1 - tpr) * k.size))]
    return float(np.mean(u >= thr))


def oscr(scores_known, correct_known, scores_unknown) -> float:
    """Open-Set Classification Rate (Dhamija et al. 2018): area under the curve of
    correct-classification rate on knowns vs. false-positive rate on unknowns."""
    sk = np.asarray(scores_known, dtype=float)
    ck = np.asarray(correct_known, dtype=bool)
    su = np.asarray(scores_unknown, dtype=float)
    if sk.size == 0 or su.size == 0:
        return float("nan")
    thresholds = np.unique(np.concatenate([sk, su]))[::-1]
    ccr = np.array([np.mean(ck & (sk >= t)) for t in thresholds])
    fpr = np.array([np.mean(su >= t) for t in thresholds])
    ccr = np.concatenate([[0.0], ccr, [ck.mean()]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    return float(np.sum((fpr[1:] - fpr[:-1]) * (ccr[1:] + ccr[:-1]) / 2))  # trapezoid rule


def risk_coverage(confidence, correct) -> tuple[np.ndarray, np.ndarray]:
    """Selective prediction curve: (coverage, risk) when rejecting least-confident first."""
    conf = np.asarray(confidence, dtype=float)
    err = 1 - np.asarray(correct, dtype=float)
    order = np.argsort(-conf, kind="mergesort")
    cum_err = np.cumsum(err[order])
    n = np.arange(1, conf.size + 1)
    return n / conf.size, cum_err / n


def aurc(confidence, correct) -> float:
    cov, risk = risk_coverage(confidence, correct)
    return float(np.mean(risk))


def risk_at_coverage(confidence, correct, coverage: float = 0.8) -> float:
    cov, risk = risk_coverage(confidence, correct)
    idx = max(0, int(np.ceil(coverage * cov.size)) - 1)
    return float(risk[idx])
