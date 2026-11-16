"""Set photo-quality thresholds from the dataset's own extreme tails (no guessing).

    python scripts/calibrate_quality.py --data-root data --out bundle/quality.json --sample 4000

Every image in the dataset is a usable photo (somebody diagnosed it), so a threshold at the dataset's
0.2 % / 99.8 % tail rejects almost no usable photo. Real farmer uploads should later be used to tighten it.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from govi_ai.inference.quality import QualityThresholds, measure  # noqa: E402
from govi_ai.manifest import load_manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=4000)
    ap.add_argument("--tail", type=float, default=0.2, help="percent of dataset images allowed per side")
    args = ap.parse_args()

    df = load_manifest(args.data_root)
    df = df[df["role"].isin(["known", "unknown_eval"])]  # plant photos only
    df = df.sample(min(args.sample, len(df)), random_state=0)
    ms = []
    for p in df["path"]:
        with Image.open(args.data_root / p) as im:
            ms.append(measure(im))
    b = np.array([m["brightness"] for m in ms])
    c = np.array([m["contrast"] for m in ms])
    s = np.array([m["sharpness"] for m in ms])
    lo, hi = args.tail, 100 - args.tail
    d = QualityThresholds()
    # Never looser than the conservative defaults, never tighter than the dataset tail.
    t = QualityThresholds(
        min_side_px=d.min_side_px,
        min_brightness=float(min(d.min_brightness, np.percentile(b, lo))),
        max_brightness=float(max(d.max_brightness, np.percentile(b, hi))),
        min_contrast=float(min(d.min_contrast, np.percentile(c, lo))),
        min_sharpness=float(min(d.min_sharpness, np.percentile(s, lo))),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t.save(args.out)
    report = {"n": len(ms), "thresholds": asdict(t),
              "dataset_percentiles": {k: dict(zip(["p0.2", "p1", "p50", "p99", "p99.8"],
                                                  np.percentile(v, [0.2, 1, 50, 99, 99.8]).round(1).tolist()))
                                      for k, v in (("brightness", b), ("contrast", c), ("sharpness", s))}}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
