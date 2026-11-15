"""Linear probes on frozen features: the phase-2 teacher experiments.

For every crop and protocol (in-domain ``split`` or ``lodo_<source>``):
  * train a class-balanced logistic-regression disease head on that crop's ``train`` rows,
    choose C on ``val`` macro-F1, fit a temperature on ``val`` NLL;
  * report closed-set metrics on ``test`` rows;
  * report open-set separation between known test images and four kinds of "unknown":
    unknown diseases of the crop, other crops, other plants, non-plant images.
The crop gate (rice | tea | chili | other_plant | non_plant) is probed on the in-domain split.

Negatives (other crops / plants / non-plant) are always taken from the in-domain ``test`` split,
which never overlaps the rows a disease head of another crop is trained on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from govi_ai.eval import metrics as M
from govi_ai.manifest import gate_label
from govi_ai.openset import Mahalanobis, energy, fit_temperature, knn_score, max_logit, msp, softmax

C_GRID = (0.01, 0.1, 1.0, 10.0)


@dataclass
class LinearHead:
    scaler: StandardScaler
    clf: LogisticRegression
    classes: list[str]
    temperature: float = 1.0

    def logits(self, X: np.ndarray) -> np.ndarray:
        """Logits in the full ``classes`` index space. Classes absent from training get a very low
        logit, so column i always means class i (sklearn only orders the classes it saw)."""
        z = self.clf.decision_function(self.scaler.transform(X))
        if z.ndim == 1:  # binary case
            z = np.stack([-z, z], 1)
        full = np.full((len(X), len(self.classes)), -1e4, dtype=np.float64)
        full[:, self.clf.classes_.astype(int)] = z
        return full


def fit_head(Xtr, ytr, Xva, yva, classes: list[str], c_grid=C_GRID, seed: int = 0) -> tuple[LinearHead, dict]:
    scaler = StandardScaler().fit(Xtr)
    Ztr, Zva = scaler.transform(Xtr), scaler.transform(Xva)
    best, best_f1, curve = None, -1.0, {}
    for c in c_grid:
        clf = LogisticRegression(C=c, max_iter=3000, class_weight="balanced", random_state=seed)
        clf.fit(Ztr, ytr)
        f1 = M.macro_f1(yva, clf.predict(Zva), len(classes)) if len(yva) else 0.0
        curve[str(c)] = f1
        if f1 > best_f1:
            best, best_f1 = clf, f1
    head = LinearHead(scaler, best, classes)
    if len(yva):
        head.temperature = fit_temperature(head.logits(Xva), yva)
    return head, {"C": float(best.C), "val_macro_f1_by_C": curve, "temperature": head.temperature}


def _open_set_block(known: dict, unk: dict) -> dict:
    """AUROC / FPR@95 for every score type, ``known`` and ``unk`` map score name -> array."""
    return {name: {"auroc": M.auroc(known[name], unk[name]), "fpr95": M.fpr_at_tpr(known[name], unk[name])}
            for name in known}


def evaluate_crop(df: pd.DataFrame, X: np.ndarray, crop: str, protocol: str, known_classes: list[str],
                  seed: int = 0) -> dict:
    """``df`` and ``X`` are row-aligned (all manifest rows of the run)."""
    is_crop_known = (df["crop"] == crop) & (df["role"] == "known")
    tr = (is_crop_known & (df[protocol] == "train")).to_numpy()
    va = (is_crop_known & (df[protocol] == "val")).to_numpy()
    te = (is_crop_known & (df[protocol] == "test")).to_numpy()
    classes = [c for c in known_classes if (df.loc[tr, "label"] == c).any()]
    if len(classes) < 2 or not te.any():
        return {"skipped": f"{len(classes)} trainable classes, {int(te.sum())} test images"}
    cidx = {c: i for i, c in enumerate(classes)}
    in_classes = df["label"].isin(classes).to_numpy()
    te = te & in_classes  # test classes never seen in training can't be scored
    va = va & in_classes  # (no in-place ops: pandas may hand out read-only arrays)
    y = df["label"].map(cidx).fillna(-1).astype(int).to_numpy()

    head, fit_info = fit_head(X[tr], y[tr], X[va], y[va], classes, seed=seed)
    lg = head.logits(X[te])
    pred = lg.argmax(1)
    p_raw = softmax(lg, 1.0)
    p_cal = softmax(lg, head.temperature)
    correct = pred == y[te]

    maha = Mahalanobis().fit(head.scaler.transform(X[tr]), y[tr])

    def scores(mask):
        z = head.logits(X[mask])
        return {"msp": msp(z, head.temperature), "max_logit": max_logit(z), "energy": energy(z),
                "mahalanobis": maha.score(head.scaler.transform(X[mask])),
                "knn": knn_score(X[tr], X[mask])}

    known_scores = scores(te)
    in_dom_test = (df["split"] == "test").to_numpy()
    unknown_sets = {
        "unknown_disease": ((df["crop"] == crop) & (df["role"] == "unknown_eval")).to_numpy(),
        "other_crops": ((df["crop"] != crop) & (df["role"] == "known")).to_numpy() & in_dom_test,
        "other_plant": (df["role"] == "other_plant").to_numpy() & in_dom_test,
        "non_plant": (df["role"] == "non_plant").to_numpy() & in_dom_test,
    }
    open_set = {}
    for name, m in unknown_sets.items():
        if m.any():
            open_set[name] = {"n": int(m.sum()), **_open_set_block(known_scores, scores(m))}
            open_set[name]["oscr_msp"] = M.oscr(known_scores["msp"], correct, scores(m)["msp"])

    per_class = M.per_class_recall(y[te], pred, len(classes))
    by_source = {}
    for s in sorted(df.loc[te, "source"].unique()):
        m = (df.loc[te, "source"] == s).to_numpy()
        by_source[s] = {"n": int(m.sum()), "macro_f1": M.macro_f1(y[te][m], pred[m], len(classes)),
                        "accuracy": M.accuracy(y[te][m], pred[m])}
    return {
        "classes": classes, "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
        "fit": fit_info,
        "closed_set": {
            "accuracy": M.accuracy(y[te], pred), "macro_f1": M.macro_f1(y[te], pred, len(classes)),
            "balanced_accuracy": M.balanced_accuracy(y[te], pred, len(classes)),
            "ece_raw": M.ece(p_raw, y[te]), "ece_calibrated": M.ece(p_cal, y[te]),
            "risk_at_80_coverage": M.risk_at_coverage(p_cal.max(1), correct, 0.8),
            "aurc": M.aurc(p_cal.max(1), correct),
            "per_class_recall": dict(zip(classes, np.nan_to_num(per_class).round(4).tolist())),
            "by_source": by_source,
        },
        "open_set": open_set,
    }


def evaluate_gate(df: pd.DataFrame, X: np.ndarray, gate_classes: list[str], seed: int = 0) -> dict:
    """Crop gate probe on the in-domain split (unknown-disease images are scored, never trained on)."""
    g = df.apply(gate_label, axis=1)
    gi = {c: i for i, c in enumerate(gate_classes)}
    y = g.map(gi).to_numpy()
    trainable = (df["role"] != "unknown_eval").to_numpy()
    tr = trainable & (df["split"] == "train").to_numpy()
    va = trainable & (df["split"] == "val").to_numpy()
    te = trainable & (df["split"] == "test").to_numpy()
    head, fit_info = fit_head(X[tr], y[tr], X[va], y[va], gate_classes, seed=seed)
    pred = head.logits(X[te]).argmax(1)
    cm = M.confusion_matrix(y[te], pred, len(gate_classes))
    unk = (df["role"] == "unknown_eval").to_numpy()
    unk_pred = head.logits(X[unk]).argmax(1) if unk.any() else np.zeros(0, int)
    crops = [c for c in gate_classes if c not in ("other_plant", "non_plant")]
    crop_idx = [gi[c] for c in crops]
    leaf_mask = np.isin(y[te], crop_idx)
    return {
        "fit": fit_info,
        "accuracy": M.accuracy(y[te], pred),
        "macro_f1": M.macro_f1(y[te], pred, len(gate_classes)),
        "per_class_recall": dict(zip(gate_classes, np.nan_to_num(M.per_class_recall(y[te], pred, len(gate_classes))).round(4).tolist())),
        "confusion": {"labels": gate_classes, "matrix": cm.tolist()},
        # wrong-crop: a rice/tea/chili leaf assigned to a different crop
        "wrong_crop_rate": float(np.mean(np.isin(pred[leaf_mask], crop_idx) & (pred[leaf_mask] != y[te][leaf_mask])))
        if leaf_mask.any() else None,
        # a crop leaf rejected as other_plant / non_plant
        "false_reject_rate": float(np.mean(~np.isin(pred[leaf_mask], crop_idx))) if leaf_mask.any() else None,
        # unknown diseases should still be routed to the right crop (so they reach the unknown check)
        "unknown_disease_right_crop": float(np.mean(unk_pred == y[unk])) if unk.any() else None,
    }
