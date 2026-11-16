"""Phases 4 + 5: calibrate thresholds, export ONNX (FP32 + INT8), check parity -> a serving bundle.

    python scripts/eval_student.py --data-root /content/data --run $GOVI/runs/mnv4_s_split   # writes outputs.npz
    python scripts/build_bundle.py --data-root /content/data --run $GOVI/runs/mnv4_s_split --out $GOVI/bundles/v1

The bundle directory is everything the service needs (see govi_ai/inference/engine.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from govi_ai.calibrate import fit_bundle  # noqa: E402
from govi_ai.export import export_onnx, exportable, quantize_int8  # noqa: E402
from govi_ai.manifest import load_manifest  # noqa: E402
from govi_ai.train.data import LabelSpace  # noqa: E402


def main() -> None:
    import torch

    from govi_ai.inference.engine import load_image, preprocess
    from govi_ai.models.student import Student

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--all-sources", action="store_true")
    ap.add_argument("--known-tpr", type=float, default=0.95)
    ap.add_argument("--target-risk", type=float, default=0.05)
    ap.add_argument("--calib-images", type=int, default=256)
    ap.add_argument("--parity-images", type=int, default=512)
    ap.add_argument("--no-int8", action="store_true")
    ap.add_argument("--yield-dir", type=Path, help="output of train_yield.py (forecasts/history/backtest)")
    ap.add_argument("--suitability-dir", type=Path, help="output of build_suitability.py")
    ap.add_argument("--quality", type=Path, help="quality.json from calibrate_quality.py")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
    space = LabelSpace(ckpt["gate_classes"], ckpt["crop_classes"])
    crops = list(ckpt["crop_classes"])
    model = Student(ckpt["timm_id"], ckpt["crop_classes"], ckpt["gate_classes"], pretrained=False,
                    scale=ckpt.get("scale", 30.0), margin=0.0)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ---- phase 4: calibration on the cached student outputs (from eval_student.py)
    df = space.encode(load_manifest(args.data_root, paper_only=not args.all_sources))
    z = np.load(args.run / "outputs.npz")
    df = df.set_index("sample_id").loc[z["sample_ids"].tolist()].reset_index()
    outs = {k: z[k] for k in z.files if k != "sample_ids"}
    heads = {c: torch.nn.functional.normalize(model.heads[c].weight, dim=1).detach().numpy() for c in crops}
    bundle, cal_report = fit_bundle(df, outs, space.gate_classes, space.crop_classes, ckpt["protocol"], heads,
                                    known_tpr=args.known_tpr, target_risk=args.target_risk, version=args.version)
    bundle.meta.update({"student": ckpt["student"], "timm_id": ckpt["timm_id"], "trained_epoch": ckpt["epoch"]})
    bundle.save(args.out)
    from govi_ai.bundle import add_extras
    from govi_ai.taxonomy import load_taxonomy

    if args.quality and args.quality.exists():
        import shutil

        shutil.copy2(args.quality, args.out / "quality.json")
    add_extras(args.out, load_taxonomy(), args.yield_dir, args.suitability_dir)
    (args.out / "preprocess.json").write_text(json.dumps(
        {"img_size": ckpt["img_size"], "mean": list(ckpt["mean"]), "std": list(ckpt["std"]), "crops": crops}, indent=2))

    # ---- phase 5: ONNX FP32 + INT8 and parity
    fp32 = args.out / "model.onnx"
    export_onnx(model, crops, ckpt["img_size"], fp32)
    rng = np.random.default_rng(0)
    val_paths = df.loc[df[ckpt["protocol"]] == "val", "path"].tolist()
    test_paths = df.loc[df["split"] == "test", "path"].tolist()

    def batch(paths, n):
        sel = [paths[i] for i in rng.choice(len(paths), min(n, len(paths)), replace=False)] if paths else []
        return [preprocess(load_image((args.data_root / p).read_bytes()), ckpt["img_size"], ckpt["mean"], ckpt["std"])
                for p in sel]

    report = {"calibration": cal_report, "files": {"model.onnx_mb": round(fp32.stat().st_size / 1e6, 2)}}
    if not args.no_int8 and val_paths:
        int8 = args.out / "model.int8.onnx"
        quantize_int8(fp32, int8, batch(val_paths, args.calib_images))
        report["files"]["model.int8.onnx_mb"] = round(int8.stat().st_size / 1e6, 2)

    import onnxruntime as ort

    xs = batch(test_paths or val_paths, args.parity_images)
    with torch.inference_mode():
        ref = [exportable(model, crops)(torch.from_numpy(x)) for x in xs]
    for name in ("model.onnx", "model.int8.onnx"):
        if not (args.out / name).exists() or not xs:
            continue
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        sess = ort.InferenceSession(str(args.out / name), opts, providers=["CPUExecutionProvider"])
        agree, diffs, times = {"gate": [], **{c: [] for c in crops}}, [], []
        for x, r in zip(xs, ref):
            t0 = time.perf_counter()
            o = sess.run(None, {"image": x})
            times.append((time.perf_counter() - t0) * 1000)
            agree["gate"].append(o[0].argmax() == r[0].numpy().argmax())
            for i, c in enumerate(crops):
                agree[c].append(o[1 + i].argmax() == r[1 + i].numpy().argmax())
            diffs.append(float(np.abs(o[0] - r[0].numpy()).max()))
        report[name] = {"argmax_agreement": {k: float(np.mean(v)) for k, v in agree.items()},
                        "gate_max_abs_diff": float(np.max(diffs)),
                        "latency_ms_1thread": {"p50": float(np.percentile(times, 50)), "p95": float(np.percentile(times, 95))}}
    (args.out / "bundle_report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps(report, indent=1, default=float))
    print(f"bundle -> {args.out}")


if __name__ == "__main__":
    main()
