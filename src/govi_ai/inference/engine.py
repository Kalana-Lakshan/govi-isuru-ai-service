"""ONNX Runtime engine: bytes in, decision out (numpy + pillow + onnxruntime only).

A *bundle* directory contains:
    model.onnx | model.int8.onnx   outputs: gate, disease_<crop>..., emb, fmap
    preprocess.json                img_size, mean, std, crops
    decision.json + prototypes_<crop>.npz   (DecisionBundle)
"""

from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from govi_ai.inference.decision import DecisionBundle

MAX_PIXELS = 40_000_000


class InvalidImage(ValueError):
    pass


def load_image(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > MAX_PIXELS:
            raise InvalidImage(f"image too large ({img.width}x{img.height})")
        img = ImageOps.exif_transpose(img)  # phone photos: respect the orientation tag
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise InvalidImage("not a readable image") from e


def preprocess(img: Image.Image, img_size: int, mean, std) -> np.ndarray:
    x = np.asarray(img.resize((img_size, img_size), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return x.transpose(2, 0, 1)[None]


def _jet(v: np.ndarray) -> np.ndarray:
    """Tiny jet colormap: v in [0,1] (H×W) -> uint8 RGB."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def cam_overlay(img: Image.Image, fmap: np.ndarray, weight: np.ndarray, size: int = 320, alpha: float = 0.45) -> str:
    """Class activation map of one class, blended over the photo; returns a base64 PNG."""
    cam = np.maximum(np.tensordot(weight, fmap, axes=([0], [0])), 0)  # h×w
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    base = img.copy()
    base.thumbnail((size, size))
    heat = Image.fromarray(_jet(cam)).resize(base.size, Image.Resampling.BICUBIC)
    out = Image.blend(base, heat, alpha)
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


class Engine:
    def __init__(self, bundle_dir: str | Path, model_file: str | None = None, threads: int = 1) -> None:
        import onnxruntime as ort

        d = Path(bundle_dir)
        pre = json.loads((d / "preprocess.json").read_text(encoding="utf-8"))
        self.img_size, self.mean, self.std = int(pre["img_size"]), pre["mean"], pre["std"]
        self.crops: list[str] = pre["crops"]
        self.decision = DecisionBundle.load(d)
        if model_file is None:
            model_file = "model.int8.onnx" if (d / "model.int8.onnx").exists() else "model.onnx"
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(d / model_file), opts, providers=["CPUExecutionProvider"])
        self.model_file = model_file
        self.output_names = [o.name for o in self.session.get_outputs()]

    def run(self, x: np.ndarray) -> dict[str, np.ndarray]:
        return dict(zip(self.output_names, self.session.run(None, {"image": x})))

    def predict(self, data: bytes, crop: str | None = None, heatmap: bool = True, return_embedding: bool = False) -> dict:
        t0 = time.perf_counter()
        img = load_image(data)
        out = self.run(preprocess(img, self.img_size, self.mean, self.std))
        disease = {c: out[f"disease_{c}"][0] for c in self.crops}
        res = self.decision.decide(out["gate"][0], disease, out["emb"][0], requested_crop=crop)
        status = res["status"]
        if heatmap and status in ("KNOWN_DISEASE", "LOW_CONFIDENCE", "UNKNOWN_DISEASE"):
            cal = self.decision.crops[res["crop"]]
            cls = res.get("class") if res.get("source") == "model" else res["top3"][0]["class"]
            if cal.head_weights is not None and cls in cal.classes:
                res["heatmap_png_base64"] = cam_overlay(img, out["fmap"][0], cal.head_weights[cal.classes.index(cls)])
        if status == "UNKNOWN_DISEASE" or return_embedding:
            # Stored by the backend so officers can review / cluster unknown cases and register
            # a confirmed new disease in the prototype library.
            res["embedding_f16_base64"] = base64.b64encode(out["emb"][0].astype(np.float16).tobytes()).decode()
        res["model"] = {"file": self.model_file, "version": self.decision.version}
        res["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return res
