"""Linear-probe experiments on cached features (phase 2). Runs on CPU in minutes.

    # one backbone
    python scripts/run_probes.py --data-root /content/data --features $GOVI/features --backbones dinov2_b
    # fusion of several frozen backbones (features are L2-normalised and concatenated)
    python scripts/run_probes.py --data-root /content/data --features $GOVI/features --backbones dinov2_b clip_b16

Writes ``<out>/probe_<backbones>.json`` and prints a summary table. By default only paper-eligible
sources are used (``use_in_paper``); ``--all-sources`` includes the rest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from govi_ai.features import load_features  # noqa: E402
from govi_ai.manifest import load_manifest, protocols  # noqa: E402
from govi_ai.probe import evaluate_crop, evaluate_gate  # noqa: E402
from govi_ai.taxonomy import load_taxonomy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--backbones", nargs="+", required=True)
    ap.add_argument("--out", type=Path, help="default: <features>/results")
    ap.add_argument("--protocols", nargs="*", help="default: split + every lodo_* column")
    ap.add_argument("--all-sources", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tax = load_taxonomy()
    df = load_manifest(args.data_root, paper_only=not args.all_sources)
    X = load_features(args.features, args.backbones, df["sample_id"].tolist())
    name = "+".join(args.backbones)
    print(f"{name}: {X.shape[0]} rows x {X.shape[1]} dims")

    results = {"backbones": args.backbones, "paper_only": not args.all_sources, "crops": {},
               "gate": evaluate_gate(df, X, tax.gate_classes, seed=args.seed)}
    protos = args.protocols or protocols(df)
    for crop in tax.crop_names:
        results["crops"][crop] = {}
        for p in protos:
            if p != "split" and not (df.loc[df[p] == "test", "crop"] == crop).any():
                continue  # this LODO column holds out another crop's source
            r = evaluate_crop(df, X, crop, p, tax.known_classes(crop), seed=args.seed)
            results["crops"][crop][p] = r
            if "skipped" in r:
                print(f"  {crop:5s} {p:32s} skipped: {r['skipped']}")
                continue
            cs, os_ = r["closed_set"], r["open_set"]
            fmt = lambda k: f"{os_[k]['energy']['auroc']:.3f}" if k in os_ else "  -  "  # noqa: E731
            print(f"  {crop:5s} {p:32s} n={r['n_test']:5d}  macroF1={cs['macro_f1']:.3f}  "
                  f"ECE={cs['ece_calibrated']:.3f}  AUROC(energy) unk={fmt('unknown_disease')} "
                  f"crops={fmt('other_crops')} plant={fmt('other_plant')} nonplant={fmt('non_plant')}")
    g = results["gate"]
    print(f"  gate  macroF1={g['macro_f1']:.3f}  wrong_crop={g['wrong_crop_rate']}  "
          f"false_reject={g['false_reject_rate']}  unknown->right_crop={g['unknown_disease_right_crop']}")

    out = args.out or args.features / "results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"probe_{name}.json"
    path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
