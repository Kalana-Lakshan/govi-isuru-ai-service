"""Compose the serving bundle from calibrated bundles of the *same* trained model, per crop, without re-running anything.

    python scripts/compose_bundle.py --base bundles/v1 --out bundle --version v1.1 \
        --novelty-from tea=bundles/v2 --coverage rice=0.7 --report bundles/v2/bundle_report.json

* ``--novelty-from CROP=DIR``  take that crop's unknown-disease score settings from another bundle (same model).
* ``--coverage CROP=SHARE``    set the LOW_CONFIDENCE threshold so the crop answers about SHARE of validation
                               photos (thresholds from ``coverage_options_val`` in ``--report``).
* ``--low-conf CROP=VALUE``    or set the threshold directly.

Only one network file is shipped: FP32 ``model.onnx`` by default (``--model int8`` for the smaller one). Every change is recorded in
``decision.json`` (meta.composition) and ``bundle_report.json``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

SKIP = {"desktop.ini", "decision.json", "bundle_report.json"}


def pairs(items: list[str]) -> dict[str, str]:
    out = {}
    for it in items or []:
        crop, _, value = it.partition("=")
        if not value:
            raise SystemExit(f"expected CROP=VALUE, got {it!r}")
        out[crop] = value
    return out


def threshold_for_coverage(report: dict, crop: str, coverage: float) -> tuple[float, dict]:
    options = report["calibration"][crop].get("coverage_options_val")
    if not options:
        raise SystemExit(f"{crop}: no coverage_options_val in the report (build it with the current build_bundle.py)")
    best = min(options, key=lambda o: abs(o["coverage"] - coverage))
    if abs(best["coverage"] - coverage) > 1e-6:
        raise SystemExit(f"{crop}: coverage {coverage} not in the report; available {[o['coverage'] for o in options]}")
    return best["threshold"], best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--novelty-from", nargs="*", default=[])
    ap.add_argument("--coverage", nargs="*", default=[])
    ap.add_argument("--low-conf", nargs="*", default=[])
    ap.add_argument("--report", type=Path, help="bundle_report.json with coverage_options_val (for --coverage)")
    ap.add_argument("--model", choices=["fp32", "int8"], default="fp32",
                    help="which network file to ship (FP32 matches the calibration exactly)")
    args = ap.parse_args()

    base = json.loads((args.base / "decision.json").read_text(encoding="utf-8"))
    base_model = json.loads((args.base / "bundle_report.json").read_text(encoding="utf-8")) \
        if (args.base / "bundle_report.json").exists() else {}
    changes = []

    for crop, src in pairs(args.novelty_from).items():
        other = json.loads((Path(src) / "decision.json").read_text(encoding="utf-8"))
        a, b = base["crops"][crop], other["crops"][crop]
        if a["classes"] != b["classes"] or abs(a["temperature"] - b["temperature"]) > 1e-9:
            raise SystemExit(f"{crop}: {src} was calibrated from a different model; refusing to mix")
        a["novelty"] = b.get("novelty", {"score": "mahalanobis"})
        changes.append({"crop": crop, "novelty": a["novelty"].get("score"), "from": str(src)})

    report = json.loads(args.report.read_text(encoding="utf-8")) if args.report else None
    for crop, cov in pairs(args.coverage).items():
        if report is None:
            raise SystemExit("--coverage needs --report")
        thr, opt = threshold_for_coverage(report, crop, float(cov))
        old = base["crops"][crop]["low_conf_threshold"]
        base["crops"][crop]["low_conf_threshold"] = thr
        changes.append({"crop": crop, "low_conf_threshold": thr, "was": old, "validation_coverage": opt["coverage"],
                        "validation_risk": opt["risk"]})
    for crop, val in pairs(args.low_conf).items():
        old = base["crops"][crop]["low_conf_threshold"]
        base["crops"][crop]["low_conf_threshold"] = float(val)
        changes.append({"crop": crop, "low_conf_threshold": float(val), "was": old})

    args.out.mkdir(parents=True, exist_ok=True)
    drop = "model.int8.onnx" if args.model == "fp32" else "model.onnx"
    for f in args.base.iterdir():
        if f.name in SKIP or f.name == drop:
            continue
        dest = args.out / f.name
        if f.is_dir():
            shutil.copytree(f, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns("desktop.ini"))
        else:
            shutil.copy2(f, dest)
    base["version"] = args.version
    base.setdefault("meta", {})["composition"] = {"base": str(args.base), "changes": changes}
    (args.out / "decision.json").write_text(json.dumps(base, indent=2), encoding="utf-8")
    rep = {**base_model, "version": args.version, "composition": base["meta"]["composition"]}
    (args.out / "bundle_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    for c in changes:
        print("changed", c)
    print(f"wrote {args.out} (version {args.version})")


if __name__ == "__main__":
    main()
