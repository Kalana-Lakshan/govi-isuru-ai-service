"""Evaluate a trained student with the same protocol as the phase-2 probes.

    python scripts/eval_student.py --data-root /content/data --run $GOVI/runs/mnv4_s_split

Runs the EMA weights over every row once (embeddings + logits cached as outputs.npz), then writes
report.json: per-crop closed-set, calibration and open-set metrics, and crop-gate metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from govi_ai.eval.report import crop_report, gate_report  # noqa: E402
from govi_ai.manifest import load_manifest  # noqa: E402
from govi_ai.train.data import LabelSpace, ManifestDataset, build_transforms  # noqa: E402


def run_model(ckpt: dict, df, data_root: Path, batch_size: int, workers: int, device: str) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from govi_ai.models.student import Student

    model = Student(ckpt["timm_id"], ckpt["crop_classes"], ckpt["gate_classes"], pretrained=False,
                    scale=ckpt.get("scale", 30.0), margin=0.0)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    ds = ManifestDataset(df, data_root, build_transforms(ckpt["img_size"], ckpt["mean"], ckpt["std"], False))
    dl = DataLoader(ds, batch_size=batch_size, num_workers=workers)
    emb, gate, dis = [], [], {c: [] for c in ckpt["crop_classes"]}
    with torch.inference_mode(), torch.autocast(device, enabled=device == "cuda"):
        for b in dl:
            out = model(b["image"].to(device))
            emb.append(out["emb"].float().cpu().numpy())
            gate.append(out["gate"].float().cpu().numpy())
            for c in dis:
                dis[c].append(out["disease"][c].float().cpu().numpy())
    return {"emb": np.concatenate(emb), "gate": np.concatenate(gate),
            **{f"disease_{c}": np.concatenate(v) for c, v in dis.items()}}


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True, help="training output dir containing best.pt")
    ap.add_argument("--all-sources", action="store_true")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
    space = LabelSpace(ckpt["gate_classes"], ckpt["crop_classes"])
    df = space.encode(load_manifest(args.data_root, paper_only=not args.all_sources))
    protocol = ckpt["protocol"]
    need = (df[protocol].isin(["train", "val", "test"]) | (df["role"] == "unknown_eval")
            | (df["split"] == "test")).to_numpy()
    df = df[need].reset_index(drop=True)

    cache = args.run / "outputs.npz"
    if cache.exists() and np.load(cache)["sample_ids"].tolist() == df["sample_id"].tolist():
        outs = dict(np.load(cache))
    else:
        outs = run_model(ckpt, df, args.data_root, args.batch_size, args.workers, device)
        np.savez(cache, sample_ids=df["sample_id"].to_numpy(dtype=str), **outs)

    report = {"protocol": protocol, "student": ckpt["student"], "epoch": ckpt["epoch"], "crops": {},
              "gate": gate_report(df, outs["gate"], space.gate_classes, df["gate_idx"].to_numpy())}
    for crop, classes in space.crop_classes.items():
        r = crop_report(df, outs["emb"], outs[f"disease_{crop}"], classes, crop, protocol)
        report["crops"][crop] = r
        if "skipped" in r:
            print(f"{crop:5s} skipped: {r['skipped']}")
            continue
        os_ = r["open_set"]
        f = lambda k: f"{os_[k]['mahalanobis']['auroc']:.3f}" if k in os_ else "  -  "  # noqa: E731
        print(f"{crop:5s} macroF1={r['closed_set']['macro_f1']:.3f} ECE={r['closed_set']['ece_calibrated']:.3f} "
              f"AUROC(maha) unk={f('unknown_disease')} crops={f('other_crops')} plant={f('other_plant')} "
              f"nonplant={f('non_plant')}")
    g = report["gate"]
    print(f"gate  macroF1={g['macro_f1']:.3f} wrong_crop={g['wrong_crop_rate']} false_reject={g['false_reject_rate']}")
    (args.run / "report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"wrote {args.run / 'report.json'}")


if __name__ == "__main__":
    main()
