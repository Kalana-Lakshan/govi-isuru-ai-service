"""Build a DEVELOPMENT bundle with random model weights, for API integration before training finishes.

    python scripts/make_dev_bundle.py --out bundle \
        --yield-dir data/yield --suitability-dir data/suitability --quality data/quality.json

The API, statuses, heatmaps, embeddings, officer prototypes, clustering, yield and suitability all work
exactly as in production, but disease predictions are MEANINGLESS (random weights). Every prediction
response carries model.warning so this can never be mistaken for a real model. Yield and suitability
are real (they don't depend on the CNN).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from govi_ai.bundle import add_extras  # noqa: E402
from govi_ai.export import export_onnx  # noqa: E402
from govi_ai.inference.decision import CropCalibration, DecisionBundle  # noqa: E402
from govi_ai.taxonomy import load_taxonomy  # noqa: E402

WARNING = "DEVELOPMENT BUNDLE: random weights, disease predictions are meaningless"


def main() -> None:
    import timm
    import torch

    from govi_ai.models.student import Student

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("bundle"))
    ap.add_argument("--student", default="mnv4_s")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--yield-dir", type=Path)
    ap.add_argument("--suitability-dir", type=Path)
    ap.add_argument("--quality", type=Path, help="quality.json from calibrate_quality.py")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tax = load_taxonomy()
    crop_classes = {c: tax.known_classes(c) for c in tax.crop_names}
    torch.manual_seed(0)
    model = Student(args.student, crop_classes, tax.gate_classes, pretrained=False, margin=0.0).eval()
    cfg = timm.data.resolve_model_data_config(model.backbone)
    crops = list(crop_classes)
    export_onnx(model, crops, args.img_size, args.out / "model.onnx")

    # Thresholds from the random model's own outputs on random inputs, so every status can occur.
    rng = np.random.default_rng(0)
    with torch.inference_mode():
        x = torch.from_numpy(rng.normal(0, 1, (64, 3, args.img_size, args.img_size)).astype(np.float32))
        out = model(x)
    emb = out["emb"].numpy().astype(np.float64)
    cov = np.cov(emb.T) + 1e-3 * np.eye(emb.shape[1])
    precision = np.linalg.inv(cov)
    cals = {}
    for c, classes in crop_classes.items():
        means = emb[rng.choice(len(emb), len(classes), replace=False)]
        cal = CropCalibration(classes=classes, temperature=1.0, means=means, precision=precision,
                              unknown_threshold=0.0, low_conf_threshold=0.0,
                              head_weights=torch.nn.functional.normalize(model.heads[c].weight, dim=1).detach().numpy())
        scores = -np.array([cal.distances(e).min() for e in emb])
        cal.unknown_threshold = float(np.quantile(scores, 0.2))
        cal.low_conf_threshold = 0.5
        cals[c] = cal
    bundle = DecisionBundle(list(tax.gate_classes), 1.0, 0.0, cals, version="dev-random",
                            meta={"warning": WARNING, "student": args.student})
    bundle.save(args.out)
    (args.out / "preprocess.json").write_text(json.dumps(
        {"img_size": args.img_size, "mean": list(cfg["mean"]), "std": list(cfg["std"]), "crops": crops}, indent=2))
    if args.quality and args.quality.exists():
        shutil.copy2(args.quality, args.out / "quality.json")
    added = add_extras(args.out, tax, args.yield_dir, args.suitability_dir)
    print(json.dumps({"bundle": str(args.out), "version": "dev-random", "added": added, "warning": WARNING}, indent=1))


if __name__ == "__main__":
    main()
