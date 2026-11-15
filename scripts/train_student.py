"""Train the student (phase 3) on a Colab GPU.

    python scripts/train_student.py --data-root /content/data --features $GOVI/features \
        --teacher dinov2_b clip_b16 --student mnv4_s --protocol split --out $GOVI/runs/mnv4_s_split

Ablations (paper): --teacher none (no KD) · --w-oe 0 (no outlier exposure) · --margin 0 (no AM-Softmax)
                   · --protocol lodo_<source> (cross-dataset)

Each run writes: config.json, teacher_<protocol>.npz (cached), history.jsonl, best.pt (EMA weights,
selected on the validation score = mean of gate macro-F1 and every crop's disease macro-F1).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from govi_ai.eval import metrics as M  # noqa: E402
from govi_ai.manifest import load_manifest  # noqa: E402
from govi_ai.taxonomy import load_taxonomy  # noqa: E402
from govi_ai.train.data import LabelSpace, ManifestDataset, balanced_weights, build_transforms, select  # noqa: E402
from govi_ai.train.losses import LossWeights, student_loss  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--features", type=Path, help="phase-2 feature dir (needed unless --teacher none)")
    ap.add_argument("--teacher", nargs="+", default=["dinov2_b", "clip_b16"], help="backbones to fuse, or 'none'")
    ap.add_argument("--student", default="mnv4_s")
    ap.add_argument("--protocol", default="split")
    ap.add_argument("--all-sources", action="store_true", help="include use_in_paper=false sources (deployment)")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--epoch-size", type=int, help="samples per epoch (default: #train rows)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--scale", type=float, default=30.0)
    ap.add_argument("--negative-share", type=float, default=0.25)
    ap.add_argument("--w-oe", type=float, default=0.5)
    ap.add_argument("--w-kd", type=float, default=1.0)
    ap.add_argument("--kd-temperature", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--max-steps", type=int, help="stop early (smoke tests)")
    ap.add_argument("--fresh", action="store_true", help="ignore last.pt and start over")
    return ap.parse_args(argv)


def teacher_targets(args, df, space):
    if args.teacher == ["none"]:
        return None
    from govi_ai.features import load_features
    from govi_ai.train.teacher import build_targets

    cache = args.out / f"teacher_{args.protocol}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if set(df["sample_id"]) <= set(z["sample_ids"].tolist()):
            return {k: z[k] for k in ("sample_ids", "gate", "disease")}
    X = load_features(args.features, args.teacher, df["sample_id"].tolist())
    t = build_targets(df, X, space, args.protocol, seed=args.seed)
    np.savez(cache, sample_ids=t["sample_ids"], gate=t["gate"], disease=t["disease"],
             info=json.dumps(t["info"], default=float))
    return t


def evaluate(model, loader, space, device) -> dict:
    import torch

    model.eval()
    gate_y, gate_p, dis = [], [], {c: ([], []) for c in space.crops}
    with torch.inference_mode(), torch.autocast(device, enabled=device == "cuda"):
        for b in loader:
            out = model(b["image"].to(device, non_blocking=True))
            gate_y.append(b["gate"].numpy())
            gate_p.append(out["gate"].float().argmax(1).cpu().numpy())
            for ci, c in enumerate(space.crops):
                m = ((b["crop_idx"] == ci) & (b["disease"] >= 0)).numpy()
                if m.any():
                    dis[c][0].append(b["disease"].numpy()[m])
                    dis[c][1].append(out["disease"][c].float().argmax(1).cpu().numpy()[m])
    res = {"gate_macro_f1": M.macro_f1(np.concatenate(gate_y), np.concatenate(gate_p), len(space.gate_classes))}
    for c, (ys, ps) in dis.items():
        if ys:
            res[f"{c}_macro_f1"] = M.macro_f1(np.concatenate(ys), np.concatenate(ps), len(space.crop_classes[c]))
    res["score"] = float(np.mean(list(res.values())))
    return res


def main(argv=None) -> None:
    import timm.utils
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from govi_ai.models.student import Student

    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tax = load_taxonomy()
    space = LabelSpace.from_taxonomy(tax)
    df = space.encode(load_manifest(args.data_root, paper_only=not args.all_sources))
    train_df, val_df = select(df, args.protocol, "train"), select(df, args.protocol, "val")
    teacher = teacher_targets(args, df, space)
    (args.out / "config.json").write_text(json.dumps(
        {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
         "gate_classes": space.gate_classes, "crop_classes": space.crop_classes,
         "n_train": len(train_df), "n_val": len(val_df)}, indent=2), encoding="utf-8")

    model = Student(args.student, space.crop_classes, space.gate_classes, pretrained=not args.no_pretrained,
                    scale=args.scale, margin=args.margin).to(device)
    cfg = timm.data.resolve_model_data_config(model.backbone)
    mean, std = cfg["mean"], cfg["std"]
    tr_ds = ManifestDataset(train_df, args.data_root, build_transforms(args.img_size, mean, std, True), teacher)
    va_ds = ManifestDataset(val_df, args.data_root, build_transforms(args.img_size, mean, std, False))
    epoch_size = args.epoch_size or len(tr_ds)
    sampler = WeightedRandomSampler(torch.as_tensor(balanced_weights(train_df, args.negative_share)),
                                    num_samples=epoch_size, replacement=True)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                       pin_memory=device == "cuda", drop_last=True, persistent_workers=args.workers > 0)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size * 2, num_workers=args.workers, pin_memory=device == "cuda")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, epoch_size // args.batch_size)
    total, warm = args.epochs * steps_per_epoch, int(args.warmup_epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) *
                                              0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    ema = timm.utils.ModelEmaV3(model, decay=args.ema)
    weights = LossWeights(oe=args.w_oe, kd=0.0 if teacher is None else args.w_kd, kd_temperature=args.kd_temperature)

    best, step, start_epoch = -1.0, 0, 0
    last = args.out / "last.pt"
    if last.exists() and not args.fresh:
        # Resume after a Colab disconnect: restore everything that changes during training.
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.module.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        best, step, start_epoch = ck["best"], ck["step"], ck["epoch"] + 1
        print(f"resumed from {last} at epoch {start_epoch} (best {best:.4f})")
    if start_epoch >= args.epochs:
        print("training already complete")
    hist = (args.out / "history.jsonl").open("a", encoding="utf-8")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, run, n = time.time(), {}, 0
        for b in tr_dl:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            with torch.autocast(device, enabled=device == "cuda"):
                out = model(b["image"])
                loss, logs = student_loss(model, out, b, weights)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            for k, v in logs.items():
                run[k] = run.get(k, 0.0) + v
            step += 1
            n += 1
            if args.max_steps and step >= args.max_steps:
                break
        n = max(1, n)
        val = evaluate(ema.module, va_dl, space, device) if len(va_ds) else {"score": 0.0}
        rec = {"epoch": epoch, "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1),
               "train": {k: round(v / n, 4) for k, v in run.items()}, "val": val}
        hist.write(json.dumps(rec) + "\n")
        hist.flush()
        print(f"epoch {epoch:3d}  loss {rec['train'].get('total', 0):.3f}  val {val}")
        if val["score"] > best:
            best = val["score"]
            torch.save({"model": ema.module.state_dict(), "student": args.student, "timm_id": model.timm_id,
                        "img_size": args.img_size, "mean": mean, "std": std, "scale": args.scale,
                        "gate_classes": space.gate_classes, "crop_classes": space.crop_classes,
                        "protocol": args.protocol, "epoch": epoch, "val": val}, args.out / "best.pt")
        # Written last and atomically, so an interrupted save never corrupts the resume point.
        tmp = last.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "ema": ema.module.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                    "step": step, "best": best}, tmp)
        tmp.replace(last)
        if args.max_steps and step >= args.max_steps:
            break
    hist.close()
    (args.out / "DONE").write_text(f"best {best:.4f}\n", encoding="utf-8")
    print(f"best val score {best:.4f} -> {args.out / 'best.pt'}")


if __name__ == "__main__":
    main()
