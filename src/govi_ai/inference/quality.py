"""Photo-quality gate (numpy only): reject photos that no model can diagnose and ask for a retake.

Measured on a 256 px greyscale copy:
  * brightness  mean intensity                 -> too dark / overexposed
  * contrast    intensity standard deviation   -> flat / covered lens
  * sharpness   variance of the Laplacian      -> motion blur / out of focus
  * size        shortest side of the original  -> thumbnail-sized uploads

Thresholds are deliberately extreme (set from the dataset's own 0.2 % / 99.8 % tails, see
scripts/calibrate_quality.py) so that a usable photo is almost never rejected; every response
carries the measured values so the thresholds can be re-tuned on real farmer uploads.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

_LAPLACE = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


@dataclass
class QualityThresholds:
    min_side_px: int = 96
    min_brightness: float = 20.0
    max_brightness: float = 245.0
    min_contrast: float = 8.0
    min_sharpness: float = 5.0

    @classmethod
    def load(cls, path: Path) -> "QualityThresholds":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8"))) if Path(path).exists() else cls()

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")


def measure(img: Image.Image) -> dict:
    g = img.convert("L")
    w, h = g.size
    scale = 256 / max(w, h)
    if scale < 1:
        g = g.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BILINEAR)
    a = np.asarray(g, dtype=np.float32)
    if min(a.shape) >= 3:
        lap = (_LAPLACE[0, 1] * a[:-2, 1:-1] + _LAPLACE[1, 0] * a[1:-1, :-2] + _LAPLACE[1, 1] * a[1:-1, 1:-1]
               + _LAPLACE[1, 2] * a[1:-1, 2:] + _LAPLACE[2, 1] * a[2:, 1:-1])
        sharp = float(lap.var())
    else:
        sharp = 0.0
    return {"min_side_px": int(min(w, h)), "brightness": round(float(a.mean()), 1),
            "contrast": round(float(a.std()), 1), "sharpness": round(sharp, 1)}


def assess(img: Image.Image, t: QualityThresholds) -> tuple[list[str], dict]:
    """Returns (problems, measurements). Empty problems = photo is usable."""
    m = measure(img)
    problems = []
    if m["min_side_px"] < t.min_side_px:
        problems.append("too_small")
    if m["brightness"] < t.min_brightness:
        problems.append("too_dark")
    if m["brightness"] > t.max_brightness:
        problems.append("overexposed")
    if m["contrast"] < t.min_contrast:
        problems.append("no_detail")
    if m["sharpness"] < t.min_sharpness:
        problems.append("blurry")
    return problems, m


ADVICE = {
    "too_small": "Use a larger photo: move closer so the leaf fills most of the picture.",
    "too_dark": "The photo is too dark. Take it in daylight or move into better light.",
    "overexposed": "The photo is too bright. Avoid direct sun on the lens or the leaf.",
    "no_detail": "The photo shows almost no detail. Check that the lens is clean and uncovered.",
    "blurry": "The photo is blurry. Hold the phone still and tap the leaf to focus.",
}
