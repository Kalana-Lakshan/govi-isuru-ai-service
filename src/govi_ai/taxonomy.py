"""Unified label space (configs/taxonomy.yaml)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from govi_ai import CONFIG_DIR

ROLES = ("known", "unknown_eval", "other_plant", "non_plant")
CLASS_TYPES = {"fungal", "bacterial", "viral", "oomycete", "algal", "pest", "abiotic", "healthy"}
_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class DiseaseClass:
    id: str
    crop: str
    type: str
    status: str  # "known" | "unknown_eval"
    en: str
    si: str = ""
    ta: str = ""


@dataclass
class Taxonomy:
    version: int
    gate_classes: list[str]
    crops: dict[str, dict[str, DiseaseClass]] = field(default_factory=dict)

    # ---- queries -------------------------------------------------------------------------------
    def has(self, crop: str, class_id: str) -> bool:
        return class_id in self.crops.get(crop, {})

    def get(self, crop: str, class_id: str) -> DiseaseClass:
        try:
            return self.crops[crop][class_id]
        except KeyError as e:
            raise KeyError(f"unknown class {crop}/{class_id}") from e

    def known_classes(self, crop: str) -> list[str]:
        """Trainable classes for a crop, in a stable (file) order -> head index order."""
        return [c.id for c in self.crops[crop].values() if c.status == "known"]

    def unknown_eval_classes(self, crop: str) -> list[str]:
        return [c.id for c in self.crops[crop].values() if c.status == "unknown_eval"]

    @property
    def crop_names(self) -> list[str]:
        return list(self.crops)


def load_taxonomy(path: str | Path | None = None) -> Taxonomy:
    path = Path(path) if path else CONFIG_DIR / "taxonomy.yaml"
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    tax = Taxonomy(version=int(raw["version"]), gate_classes=list(raw["gate_classes"]))

    for crop, spec in raw["crops"].items():
        if crop not in tax.gate_classes:
            raise ValueError(f"crop '{crop}' is missing from gate_classes")
        classes: dict[str, DiseaseClass] = {}
        for cid, c in spec["classes"].items():
            if not _ID_RE.match(cid):
                raise ValueError(f"bad class id '{cid}' (use snake_case)")
            if c.get("type") not in CLASS_TYPES:
                raise ValueError(f"{crop}/{cid}: type must be one of {sorted(CLASS_TYPES)}")
            if c.get("status") not in ("known", "unknown_eval"):
                raise ValueError(f"{crop}/{cid}: status must be known|unknown_eval")
            classes[cid] = DiseaseClass(
                id=cid, crop=crop, type=c["type"], status=c["status"], en=c.get("en", cid),
                si=c.get("si") or "", ta=c.get("ta") or "",
            )
        if "healthy" not in classes:
            raise ValueError(f"crop '{crop}' must define a 'healthy' class")
        tax.crops[crop] = classes

    for g in ("other_plant", "non_plant"):
        if g not in tax.gate_classes:
            raise ValueError(f"gate_classes must include '{g}'")
    return tax
