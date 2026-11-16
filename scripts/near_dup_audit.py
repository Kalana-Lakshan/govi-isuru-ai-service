"""Embedding-based near-duplicate audit of every test set against its training set (paper analysis).

Perceptual hashing misses crops, zooms and re-shot copies; self-supervised embeddings catch them
(guava audit: pHash AUC 0.53 vs embedding detector 0.985). For each protocol (in-domain `split` and
each `lodo_*`), every test image gets its maximum cosine similarity to any training image in DINOv2
feature space. Test images at or above ``--threshold`` are written to an exclusion list; evaluation
then also reports metrics on the remaining "clean" test set (eval_student.py --exclude-file).

    python scripts/near_dup_audit.py --data-root /content/data --features $GOVI/features --out $GOVI/audit

Outputs: near_dup_scores.csv (every test row), near_dup_exclusions.csv (flagged rows), summary.json, and
montages (pairs_top.png, pairs_at_threshold.png) to check by eye that the threshold is right.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from govi_ai.features import load_features  # noqa: E402
from govi_ai.manifest import load_manifest, protocols  # noqa: E402

GRID = (0.90, 0.93, 0.95, 0.97, 0.99)


def max_similarity(test: np.ndarray, train: np.ndarray, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    best = np.full(len(test), -1.0, dtype=np.float32)
    arg = np.zeros(len(test), dtype=np.int64)
    for i in range(0, len(test), chunk):
        s = test[i:i + chunk] @ train.T
        arg[i:i + chunk] = s.argmax(1)
        best[i:i + chunk] = s.max(1)
    return best, arg


def montage(data_root: Path, pairs: list[tuple[str, str, float]], out: Path, size: int = 120) -> None:
    if not pairs:
        return
    cols = 4  # 4 pairs per row
    rows = (len(pairs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * (2 * size + 10), rows * (size + 18)), "white")
    d = ImageDraw.Draw(canvas)
    for k, (a, b, s) in enumerate(pairs):
        x, y = (k % cols) * (2 * size + 10), (k // cols) * (size + 18)
        for j, p in enumerate((a, b)):
            with Image.open(data_root / p) as im:
                canvas.paste(im.convert("RGB").resize((size, size)), (x + j * size, y + 16))
        d.text((x + 2, y + 2), f"cos={s:.3f}", fill="black")
    canvas.save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--backbone", default="dinov2_b")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = load_manifest(args.data_root)
    X = load_features(args.features, [args.backbone], df["sample_id"].tolist())  # L2-normalised
    rows, summary = [], {"backbone": args.backbone, "threshold": args.threshold, "protocols": {}}
    for p in protocols(df):
        trainable = (df["role"] != "unknown_eval").to_numpy()
        tr = np.flatnonzero(trainable & (df[p] == "train").to_numpy())
        te = np.flatnonzero((df[p] == "test").to_numpy() | ((df["role"] == "unknown_eval") & (df[p] == "unknown_eval")).to_numpy())
        if not len(tr) or not len(te):
            continue
        s, a = max_similarity(X[te], X[tr])
        for i, (row, score, nn) in enumerate(zip(te, s, a)):
            rows.append({"protocol": p, "sample_id": df.at[row, "sample_id"], "role": df.at[row, "role"],
                         "max_sim": round(float(score), 4), "nearest_train_id": df.at[tr[nn], "sample_id"],
                         "same_label": df.at[row, "label"] == df.at[tr[nn], "label"]})
        is_test = df.loc[te, "role"].to_numpy() != "unknown_eval"
        summary["protocols"][p] = {
            "n_test": int(is_test.sum()), "n_unknown_eval": int((~is_test).sum()),
            "share_test_at_or_above": {str(t): round(float((s[is_test] >= t).mean()), 4) for t in GRID},
            "share_unknown_at_or_above": {str(t): round(float((s[~is_test] >= t).mean()), 4) for t in GRID}
            if (~is_test).any() else {},
            "median_test_max_sim": round(float(np.median(s[is_test])), 4) if is_test.any() else None,
        }
        if p == "split":
            order = np.argsort(-s)
            top = [(df.at[te[i], "path"], df.at[tr[a[i]], "path"], float(s[i])) for i in order[:24]]
            near = np.argsort(np.abs(s - args.threshold))[:24]
            at = [(df.at[te[i], "path"], df.at[tr[a[i]], "path"], float(s[i])) for i in near]
            montage(args.data_root, top, args.out / "pairs_top.png")
            montage(args.data_root, at, args.out / "pairs_at_threshold.png")
        print(p, summary["protocols"][p]["share_test_at_or_above"])

    with (args.out / "near_dup_scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    flagged = [r for r in rows if r["max_sim"] >= args.threshold]
    with (args.out / "near_dup_exclusions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["protocol", "sample_id", "max_sim"])
        w.writeheader()
        w.writerows({k: r[k] for k in ("protocol", "sample_id", "max_sim")} for r in flagged)
    summary["flagged_rows"] = len(flagged)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"{len(flagged)} test rows flagged at cos >= {args.threshold} -> {args.out}")


if __name__ == "__main__":
    main()
