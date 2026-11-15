"""Leakage-safe splits.

Two protocols are written into the manifest:

* ``split`` (in-domain): train / val / test = 70 / 15 / 15, stratified by (crop, label) for disease
  images and by (role, source) for negatives, *grouped by near-duplicate cluster* so no photo (or
  augmented copy of it) is on both sides. ``unknown_eval`` images get ``split = unknown_eval``:
  they are never trained on and never used for threshold tuning.

* ``lodo_<source>`` (leave-one-dataset-out, the cross-domain protocol for the paper): the held-out
  source is ``test``; all other sources of the same crop are split train / val (85 / 15);
  rows of other crops and negatives keep their in-domain split; any row whose near-duplicate group
  also occurs in the held-out source is ``excluded``.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

N_FOLDS = 20  # 1 fold = 5 %
TEST_FOLDS = (0, 1, 2)
VAL_FOLDS = (3, 4, 5)


def _fold_assign(strata: list[str], groups: list[str], seed: int) -> np.ndarray:
    """Return a fold index (0..N_FOLDS-1) per row; rows of one group share a fold."""
    n = len(strata)
    folds = np.full(n, -1, dtype=int)
    if n == 0:
        return folds
    counts = Counter(strata)
    # StratifiedGroupKFold needs >= n_splits members per class; rare strata go to train.
    common = np.array([counts[s] >= N_FOLDS for s in strata])
    if common.sum() >= N_FOLDS:
        idx = np.flatnonzero(common)
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for f, (_, test_idx) in enumerate(sgkf.split(idx, [strata[i] for i in idx], [groups[i] for i in idx])):
            folds[idx[test_idx]] = f
    folds[folds == -1] = N_FOLDS - 1  # train
    # A group may straddle "common" and "rare" rows: force one fold per group.
    first: dict[str, int] = {}
    for i, g in enumerate(groups):
        folds[i] = first.setdefault(g, folds[i])
    return folds


def _fold_to_split(f: int) -> str:
    if f in TEST_FOLDS:
        return "test"
    if f in VAL_FOLDS:
        return "val"
    return "train"


def in_domain_split(rows: list[dict], seed: int = 42) -> None:
    """Write ``row['split']`` in place."""
    trainable = [r for r in rows if r["role"] != "unknown_eval"]
    strata = [f"{r['crop']}:{r['label']}" if r["role"] == "known" else f"{r['role']}:{r['source']}"
              for r in trainable]
    folds = _fold_assign(strata, [r["group_id"] for r in trainable], seed)
    for r, f in zip(trainable, folds):
        r["split"] = _fold_to_split(int(f))
    for r in rows:
        if r["role"] == "unknown_eval":
            r["split"] = "unknown_eval"


def lodo_split(rows: list[dict], held_source: str, seed: int = 42) -> list[str]:
    """Leave-one-dataset-out assignment for ``held_source`` (requires ``split`` already set)."""
    held = [r for r in rows if r["source"] == held_source]
    if not held:
        raise ValueError(f"no rows for source {held_source}")
    crop = held[0]["crop"]
    held_groups = {r["group_id"] for r in held}

    out = ["excluded"] * len(rows)
    same_crop_idx: list[int] = []
    for i, r in enumerate(rows):
        if r["source"] == held_source:
            out[i] = "test" if r["role"] == "known" else "unknown_eval"
        elif r["group_id"] in held_groups:
            out[i] = "excluded"
        elif r["crop"] == crop and r["role"] == "known":
            same_crop_idx.append(i)
        else:
            out[i] = r["split"] if r["split"] != "test" else "train"

    # Re-split the remaining same-crop sources into train/val (85/15) by group.
    sub = [rows[i] for i in same_crop_idx]
    folds = _fold_assign([r["label"] for r in sub], [r["group_id"] for r in sub], seed)
    for i, f in zip(same_crop_idx, folds):
        out[i] = "val" if int(f) in TEST_FOLDS else "train"
    return out


def add_lodo_columns(rows: list[dict], sources: list[str], min_images: int = 200, seed: int = 42) -> list[str]:
    """Add ``lodo_<source>`` columns for every disease source with >= ``min_images`` known rows."""
    counts = Counter(r["source"] for r in rows if r["role"] == "known")
    made = []
    for s in sources:
        if counts.get(s, 0) < min_images:
            continue
        col = f"lodo_{s}"
        for r, v in zip(rows, lodo_split(rows, s, seed)):
            r[col] = v
        made.append(col)
    return made
