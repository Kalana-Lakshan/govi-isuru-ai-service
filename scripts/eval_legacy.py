"""Baseline: evaluate the Govi Isuru v1 Keras models on the new leakage-safe test sets.

These numbers are the "before" column of the paper. Run on Colab:

    pip install -e ".[legacy]"
    python scripts/eval_legacy.py --data-root /content/drive/MyDrive/govi-ai/data \
        --legacy-models /content/drive/MyDrive/govi-ai/legacy_models --out artifacts/legacy_baseline.json

``--legacy-models`` must contain the v1 layout:
    best_model.keras, class_indices.json, tea/tea_best_model.keras, tea/tea_class_indices.json,
    chili/chili_best_model.keras, chili/chili_class_indices.json

Reported per crop:
  * closed-set metrics on the in-domain test split, restricted to classes the v1 model knows
    (also broken down by source; sources the v1 model was trained on are flagged ``seen_by_v1``)
  * share of test images whose true class the v1 model cannot output at all
  * open-set AUROC / FPR@95 using max softmax probability for: unknown diseases of that crop,
    other crops' leaves, other plants and non-plant images
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from govi_ai.eval import metrics as M  # noqa: E402

LEGACY = {
    "rice": {
        "model": "best_model.keras", "classes": "class_indices.json",
        "map": {"Bacterial Leaf Blight": "bacterial_leaf_blight", "Brown Spot": "brown_spot",
                "Healthy Rice Leaf": "healthy", "Leaf Blast": "blast", "Leaf scald": "leaf_scald",
                "Narrow Brown Leaf Spot": "narrow_brown_leaf_spot", "Rice Hispa": "hispa",
                "Sheath Blight": "sheath_blight"},
        "seen_sources": ["rice_rlbf"],
    },
    "tea": {
        "model": "tea/tea_best_model.keras", "classes": "tea/tea_class_indices.json",
        "map": {"Blister Blight": "blister_blight", "Brown Blight": "brown_blight", "Gray Blight": "gray_blight",
                "Healthy": "healthy", "Red Rust": "red_rust"},
        "seen_sources": ["legacy_tea"],
    },
    "chili": {
        "model": "chili/chili_best_model.keras", "classes": "chili/chili_class_indices.json",
        "map": {"Healthy": "healthy", "Leaf Spot": "leaf_spot", "Thrips Damage": "mites_thrips",
                "Yellow Virus": "yellow_leaf_virus"},
        "seen_sources": ["legacy_chili"],
    },
}


def legacy_class_names(path: Path) -> list[str]:
    """v1 stored rice as {name: idx} and tea/chili as {idx: name}."""
    d = json.loads(path.read_text(encoding="utf-8"))
    if all(str(k).isdigit() for k in d):
        return [d[k] for k in sorted(d, key=int)]
    return [k for k, _ in sorted(d.items(), key=lambda kv: kv[1])]


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0  # v1 preprocessing: rescale only


def predict(model, paths: list[Path], batch: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(paths), batch):
        x = np.stack([load_image(p) for p in paths[i:i + batch]])
        out.append(model.predict(x, verbose=0))
    return np.concatenate(out) if out else np.zeros((0, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--legacy-models", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/legacy_baseline.json"))
    ap.add_argument("--max-ood", type=int, default=1500, help="cap per OOD group (speed)")
    args = ap.parse_args()

    from tensorflow import keras  # only needed here

    with (args.data_root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rng = np.random.default_rng(0)
    results: dict = {}

    for crop, cfg in LEGACY.items():
        model = keras.models.load_model(args.legacy_models / cfg["model"])
        names = legacy_class_names(args.legacy_models / cfg["classes"])
        unified = [cfg["map"][n] for n in names]
        idx_of = {u: i for i, u in enumerate(unified)}

        test = [r for r in rows if r["crop"] == crop and r["role"] == "known" and r["split"] == "test"]
        supported = [r for r in test if r["label"] in idx_of]
        probs = predict(model, [args.data_root / r["path"] for r in supported])
        y = np.array([idx_of[r["label"]] for r in supported], dtype=int)
        pred = probs.argmax(1) if len(supported) else np.zeros(0, dtype=int)
        msp_known = probs.max(1) if len(supported) else np.zeros(0)

        by_source = {}
        for s in sorted({r["source"] for r in supported}):
            m = np.array([r["source"] == s for r in supported])
            by_source[s] = {
                "n": int(m.sum()), "seen_by_v1": s in cfg["seen_sources"],
                "accuracy": M.accuracy(y[m], pred[m]),
                "macro_f1": M.macro_f1(y[m], pred[m], len(unified)),
            }

        def ood(sel):
            pool = [r for r in rows if sel(r) and r["split"] in ("test", "unknown_eval")]
            if len(pool) > args.max_ood:
                pool = [pool[i] for i in rng.choice(len(pool), args.max_ood, replace=False)]
            if not pool:
                return None
            s = predict(model, [args.data_root / r["path"] for r in pool]).max(1)
            return {"n": len(pool), "auroc": M.auroc(msp_known, s), "fpr95": M.fpr_at_tpr(msp_known, s)}

        results[crop] = {
            "v1_classes": unified,
            "test_images": len(test),
            "unsupported_share": 1 - len(supported) / max(1, len(test)),
            "closed_set": {
                "n": len(supported),
                "accuracy": M.accuracy(y, pred) if len(y) else None,
                "macro_f1": M.macro_f1(y, pred, len(unified)) if len(y) else None,
                "balanced_accuracy": M.balanced_accuracy(y, pred, len(unified)) if len(y) else None,
                "ece": M.ece(probs, y) if len(y) else None,
                "per_class_recall": dict(zip(unified, M.per_class_recall(y, pred, len(unified)).tolist())),
                "by_source": by_source,
            },
            "open_set_msp": {
                "unknown_disease": ood(lambda r: r["crop"] == crop and r["role"] == "unknown_eval"),
                "other_crops": ood(lambda r: r["crop"] not in ("", crop) and r["role"] == "known"),
                "other_plant": ood(lambda r: r["role"] == "other_plant"),
                "non_plant": ood(lambda r: r["role"] == "non_plant"),
            },
        }
        print(json.dumps({crop: results[crop]["closed_set"] | {"open_set": results[crop]["open_set_msp"]}},
                         indent=1, default=float))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
