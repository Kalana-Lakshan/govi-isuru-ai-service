"""Explainability for the paper: CAM (what the service serves) vs Grad-CAM vs Grad-CAM++, with a
quantitative faithfulness check. Offline only (needs PyTorch); never part of the serving image.

Faithfulness = deletion test: pixels are removed (replaced by a blurred copy) in order of saliency,
5 % at a time up to 50 %, and the predicted class probability is tracked. A faithful map makes the
probability fall fast, i.e. a low area under the deletion curve (AUC). A random map is the baseline.

    python scripts/xai_figures.py --data-root /content/data --run $GOVI/runs/mnv4_s_split --out $GOVI/xai
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from govi_ai.inference.engine import _jet  # noqa: E402
from govi_ai.manifest import load_manifest  # noqa: E402

FRACTIONS = np.linspace(0, 0.5, 11)
METHODS = ("cam", "gradcam", "gradcam_pp", "random")


def saliency_maps(model, x, crop: str, cls: int) -> dict[str, np.ndarray]:
    """All maps at feature-map resolution, normalised to [0, 1]."""
    import torch

    model.zero_grad(set_to_none=True)
    fmap = model.backbone.forward_features(x)
    fmap.retain_grad()
    logit = model.heads[crop](fmap.mean((2, 3)))[0, cls]
    logit.backward()
    a, g = fmap.detach()[0], fmap.grad.detach()[0]           # C×h×w
    out = {}
    with torch.no_grad():
        out["cam"] = model.cam(fmap.detach(), crop, torch.tensor([cls]))[0]
        w = g.mean((1, 2))
        out["gradcam"] = torch.relu((w[:, None, None] * a).sum(0))
        g2, g3 = g ** 2, g ** 3
        denom = 2 * g2 + (a * g3).sum((1, 2), keepdim=True)
        alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
        w_pp = (alpha * torch.relu(g)).sum((1, 2))
        out["gradcam_pp"] = torch.relu((w_pp[:, None, None] * a).sum(0))
    res = {}
    for k, v in out.items():
        v = v.float().cpu().numpy()
        res[k] = (v - v.min()) / (v.max() - v.min() + 1e-8)
    res["random"] = np.random.default_rng(cls).random(res["cam"].shape)
    return res


def deletion_auc(model, img: Image.Image, sal: np.ndarray, crop: str, cls: int, tf) -> tuple[float, list[float]]:
    import torch

    size = img.size
    s = np.asarray(Image.fromarray((sal * 255).astype(np.uint8)).resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
    order = np.argsort(-s.ravel())
    base = np.asarray(img, dtype=np.uint8).copy()
    blurred = np.asarray(img.filter(ImageFilter.GaussianBlur(radius=max(size) / 30)), dtype=np.uint8)
    probs = []
    for f in FRACTIONS:
        cur = base.copy().reshape(-1, 3)
        k = int(f * len(order))
        cur[order[:k]] = blurred.reshape(-1, 3)[order[:k]]
        x = tf(Image.fromarray(cur.reshape(base.shape)))[None]
        with torch.no_grad():
            p = torch.softmax(model(x)["disease"][crop], 1)[0, cls].item()
        probs.append(p)
    p = np.array(probs) / max(probs[0], 1e-8)  # relative to the unmodified image
    return float(np.sum((p[1:] + p[:-1]) / 2 * np.diff(FRACTIONS)) / FRACTIONS[-1]), probs


def overlay(img: Image.Image, sal: np.ndarray, size: int = 180) -> Image.Image:
    base = img.resize((size, size))
    heat = Image.fromarray(_jet(sal)).resize((size, size), Image.Resampling.BICUBIC)
    return Image.blend(base, heat, 0.45)


def main() -> None:
    import torch

    from govi_ai.models.student import Student
    from govi_ai.train.data import build_transforms

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-crop", type=int, default=60, help="test images per crop for faithfulness")
    ap.add_argument("--figure-rows", type=int, default=6)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
    model = Student(ck["timm_id"], ck["crop_classes"], ck["gate_classes"], pretrained=False, margin=0.0)
    model.load_state_dict(ck["model"])
    model.eval()
    tf = build_transforms(ck["img_size"], ck["mean"], ck["std"], train=False)
    df = load_manifest(args.data_root)
    rng = np.random.default_rng(0)
    report: dict = {"run": str(args.run), "fractions": FRACTIONS.tolist(), "crops": {}}
    for crop, classes in ck["crop_classes"].items():
        te = df[(df["crop"] == crop) & (df["role"] == "known") & (df[ck["protocol"]] == "test")
                & df["label"].isin(classes)]
        if te.empty:
            continue
        te = te.iloc[rng.permutation(len(te))[: args.per_crop]]
        aucs = {m: [] for m in METHODS}
        rows = []
        for _, r in te.iterrows():
            with Image.open(args.data_root / r["path"]) as im:
                img = im.convert("RGB")
            x = tf(img)[None]
            with torch.no_grad():
                cls = int(model(x)["disease"][crop].argmax(1))
            maps = saliency_maps(model, x.requires_grad_(False), crop, cls)
            for m in METHODS:
                aucs[m].append(deletion_auc(model, img, maps[m], crop, cls, tf)[0])
            if len(rows) < args.figure_rows:
                rows.append((img, maps, classes[cls], r["label"]))
        report["crops"][crop] = {"n": len(te), "deletion_auc_mean": {m: round(float(np.mean(v)), 4) for m, v in aucs.items()},
                                 "deletion_auc_std": {m: round(float(np.std(v)), 4) for m, v in aucs.items()}}
        # figure: image | CAM | Grad-CAM | Grad-CAM++
        S = 180
        fig = Image.new("RGB", (4 * S, len(rows) * (S + 16) + 16), "white")
        d = ImageDraw.Draw(fig)
        for j, t in enumerate(("photo", "CAM (served)", "Grad-CAM", "Grad-CAM++")):
            d.text((j * S + 4, 2), t, fill="black")
        for i, (img, maps, pred, true) in enumerate(rows):
            y = 16 + i * (S + 16)
            fig.paste(img.resize((S, S)), (0, y + 14))
            for j, m in enumerate(("cam", "gradcam", "gradcam_pp")):
                fig.paste(overlay(img, maps[m], S), ((j + 1) * S, y + 14))
            d.text((4, y + 1), f"pred {pred} / true {true}", fill="black")
        fig.save(args.out / f"xai_{crop}.png")
        print(crop, report["crops"][crop]["deletion_auc_mean"])
    (args.out / "faithfulness.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
