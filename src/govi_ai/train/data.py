"""Datasets, augmentation and sampling for the student."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from govi_ai.manifest import gate_label
from govi_ai.taxonomy import Taxonomy

PAD = -1e4  # teacher logit for classes the teacher never saw


@dataclass
class LabelSpace:
    gate_classes: list[str]
    crop_classes: dict[str, list[str]]

    @classmethod
    def from_taxonomy(cls, tax: Taxonomy) -> "LabelSpace":
        return cls(list(tax.gate_classes), {c: tax.known_classes(c) for c in tax.crop_names})

    @property
    def crops(self) -> list[str]:
        return list(self.crop_classes)

    @property
    def max_classes(self) -> int:
        return max(len(v) for v in self.crop_classes.values())

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add integer targets: gate_idx, crop_idx (-1 = not a crop image), disease_idx (-1 = none)."""
        out = df.copy()
        gi = {c: i for i, c in enumerate(self.gate_classes)}
        ci = {c: i for i, c in enumerate(self.crops)}
        out["gate_idx"] = out.apply(gate_label, axis=1).map(gi).astype(int)
        out["crop_idx"] = out["crop"].map(ci).fillna(-1).astype(int)
        di = []
        for crop, label, role in zip(out["crop"], out["label"], out["role"]):
            classes = self.crop_classes.get(crop, [])
            di.append(classes.index(label) if role == "known" and label in classes else -1)
        out["disease_idx"] = di
        return out


def select(df: pd.DataFrame, protocol: str, part: str) -> pd.DataFrame:
    """Rows of ``part`` (train|val|test) under ``protocol``; unknown-disease rows are never trainable."""
    m = (df[protocol] == part) & (df["role"] != "unknown_eval")
    return df[m].reset_index(drop=True)


def balanced_weights(df: pd.DataFrame, negative_share: float = 0.25) -> np.ndarray:
    """Sampling weights: every known (crop, class) equally likely; negatives (other_plant / non_plant)
    together get ``negative_share`` of the draws, split equally per source."""
    w = np.zeros(len(df))
    known = (df["role"] == "known").to_numpy()
    neg = ~known
    if known.any():
        keys = df.loc[known, "crop"] + "/" + df.loc[known, "label"]
        counts = keys.map(keys.value_counts())
        w[known] = (1 - negative_share) / keys.nunique() / counts.to_numpy()
    if neg.any():
        src = df.loc[neg, "source"]
        w[neg] = negative_share / src.nunique() / src.map(src.value_counts()).to_numpy()
    if not neg.any() or not known.any():
        w = w / w.sum()
    return w


def build_transforms(img_size: int, mean, std, train: bool):
    from torchvision import transforms as T

    if not train:
        return T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(), T.Normalize(mean, std),
        ])
    # Strong, geometry-heavy augmentation: the only remedy with consistent cross-dataset gains in
    # rice-disease transfer (Paul 2026). Crops keep >= 35 % of the image so lesions usually survive.
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.35, 1.0), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomApply([T.RandomRotation(30)], p=0.5),
        T.TrivialAugmentWide(),
        T.ToTensor(), T.Normalize(mean, std),
        T.RandomErasing(p=0.25, value="random"),
    ])


class ManifestDataset:
    """Yields dicts: image, gate, crop_idx, disease and (if available) teacher logits."""

    def __init__(self, df: pd.DataFrame, data_root: Path, transform, teacher: dict | None = None,
                 n_gate: int = 5, k_max: int = 1) -> None:
        self.df = df.reset_index(drop=True)
        self.root, self.tf = Path(data_root), transform
        self.gate = self.df["gate_idx"].to_numpy()
        self.crop = self.df["crop_idx"].to_numpy()
        self.disease = self.df["disease_idx"].to_numpy()
        self.paths = self.df["path"].tolist()
        self.teacher = None
        if teacher is not None:
            idx = {s: i for i, s in enumerate(teacher["sample_ids"])}
            rows = [idx.get(s, -1) for s in self.df["sample_id"]]
            if min(rows, default=0) < 0:
                raise KeyError("teacher targets are missing some training rows; rebuild them for this protocol")
            self.teacher = {"gate": teacher["gate"][rows].astype(np.float32),
                            "disease": teacher["disease"][rows].astype(np.float32)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        import torch

        with Image.open(self.root / self.paths[i]) as im:
            x = self.tf(im.convert("RGB"))
        item = {"image": x, "gate": torch.tensor(self.gate[i]), "crop_idx": torch.tensor(self.crop[i]),
                "disease": torch.tensor(self.disease[i])}
        if self.teacher is not None:
            item["t_gate"] = torch.from_numpy(self.teacher["gate"][i])
            item["t_disease"] = torch.from_numpy(self.teacher["disease"][i])
        return item
