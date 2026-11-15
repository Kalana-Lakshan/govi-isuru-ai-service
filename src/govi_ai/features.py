"""Extract and store frozen-backbone features for every manifest row.

Features are computed once per backbone and cached as ``<out>/<key>.npz`` (float16 ``feats`` +
``sample_ids``), so every probe, fusion and open-set experiment afterwards runs in seconds on CPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image


def _transform(img_size: int, mean, std):
    from torchvision import transforms as T

    # Whole-image resize (no centre crop): lesions near the leaf edge must stay visible.
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


class _ImageList:
    def __init__(self, paths: Sequence[Path], tf) -> None:
        self.paths, self.tf = list(paths), tf

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB"))


def extract(model, paths: Sequence[Path], mean, std, *, img_size: int = 224, batch_size: int = 64,
            num_workers: int = 2, device: str | None = None, hflip_tta: bool = True) -> np.ndarray:
    """Pooled features for ``paths`` (float16). With ``hflip_tta`` the features of the image and
    its mirror are averaged (cheap, removes left/right orientation bias)."""
    import torch
    from torch.utils.data import DataLoader

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loader = DataLoader(_ImageList(paths, _transform(img_size, mean, std)), batch_size=batch_size,
                        num_workers=num_workers, pin_memory=device == "cuda")
    out = []
    amp = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else torch.autocast("cpu", enabled=False)
    with torch.inference_mode(), amp:
        for x in loader:
            x = x.to(device, non_blocking=True)
            f = model(x)
            if hflip_tta:
                f = (f + model(torch.flip(x, dims=[3]))) / 2
            out.append(f.float().cpu().numpy().astype(np.float16))
    return np.concatenate(out) if out else np.zeros((0, 0), np.float16)


def save_features(path: Path, feats: np.ndarray, sample_ids: Sequence[str], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, feats=feats, sample_ids=np.asarray(sample_ids, dtype=str), **{f"meta_{k}": v for k, v in meta.items()})


def load_features(feature_dir: Path, keys: Sequence[str], sample_ids: Sequence[str]) -> np.ndarray:
    """Load one or more backbones' features aligned to ``sample_ids``. Each backbone's features are
    L2-normalised before concatenation, so every backbone contributes equally to a fusion."""
    parts = []
    for k in keys:
        z = np.load(Path(feature_dir) / f"{k}.npz")
        index = {s: i for i, s in enumerate(z["sample_ids"].tolist())}
        missing = [s for s in sample_ids if s not in index]
        if missing:
            raise KeyError(f"{k}: {len(missing)} sample ids have no features (re-run extraction)")
        f = z["feats"][[index[s] for s in sample_ids]].astype(np.float32)
        parts.append(f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8))
    return np.concatenate(parts, axis=1)
