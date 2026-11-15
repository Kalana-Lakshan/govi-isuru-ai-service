"""Extract frozen-backbone features for every manifest row (phase 2, run on a Colab GPU).

    python scripts/extract_features.py --data-root /content/data --out /content/drive/MyDrive/govi-ai/features \
        --backbones dinov2_b clip_b16 effnet_b0

Writes ``<out>/<backbone>.npz``. Re-running skips backbones whose file already covers every row.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from govi_ai.features import extract, save_features  # noqa: E402
from govi_ai.manifest import load_manifest  # noqa: E402
from govi_ai.models.backbones import BACKBONES, create_backbone  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backbones", nargs="+", default=["dinov2_b", "clip_b16", "effnet_b0"],
                    help=f"keys from {sorted(BACKBONES)} or raw timm ids")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="first N rows only (smoke test)")
    ap.add_argument("--no-tta", action="store_true", help="disable horizontal-flip averaging")
    args = ap.parse_args()

    df = load_manifest(args.data_root)
    if args.limit:
        df = df.head(args.limit)
    ids = df["sample_id"].tolist()
    paths = [args.data_root / p for p in df["path"]]

    for key in args.backbones:
        out = args.out / f"{key}.npz"
        if out.exists():
            have = set(np.load(out)["sample_ids"].tolist())
            if have.issuperset(ids):
                print(f"[skip] {key}: features cover all {len(ids)} rows")
                continue
        t0 = time.time()
        model, mean, std = create_backbone(key, img_size=args.img_size)
        feats = extract(model, paths, mean, std, img_size=args.img_size, batch_size=args.batch_size,
                        num_workers=args.workers, hflip_tta=not args.no_tta)
        save_features(out, feats, ids, {"backbone": key, "img_size": args.img_size, "hflip_tta": not args.no_tta})
        dt = time.time() - t0
        print(f"[ok]   {key}: {feats.shape} in {dt / 60:.1f} min ({len(ids) / max(dt, 1e-6):.0f} img/s) -> {out}")
        del model


if __name__ == "__main__":
    main()
