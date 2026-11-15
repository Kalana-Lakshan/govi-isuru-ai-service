"""Dataset source registry (configs/datasets.yaml) and label resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from govi_ai import CONFIG_DIR
from govi_ai.taxonomy import Taxonomy

KEEP = "@keep"            # other_plant / non_plant: keep every label
AS_NON_PLANT = "@non_plant"  # inside an other_plant source: this label is a non-plant image
WILDCARD = "*"
FETCH_METHODS = {"hf", "manual", "local", "coco"}


def norm_label(s: str) -> str:
    """Case/spacing-insensitive key: 'Leaf_Red Rust' == 'leaf-red-rust' == 'leafredrust'."""
    return re.sub(r"[\s_\-]+", "", str(s)).lower()


class UnmappedLabelError(KeyError):
    """A source label has no entry in label_map. Never silently drop data."""


@dataclass(frozen=True)
class Resolved:
    role: str              # known | unknown_eval | other_plant | non_plant
    crop: str | None       # rice | tea | chili | None
    label: str             # unified class id, or "<source>:<raw label>" for negatives


@dataclass
class Source:
    id: str
    crop: str | None
    role: str
    fetch: dict[str, Any]
    license: str
    citation: str
    use_in_paper: bool
    label_map: dict[str, str | None]
    enabled: bool = True
    train_only: bool = False
    max_per_class: int | None = None
    origin: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    _norm_map: dict[str, str | None] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._norm_map = {norm_label(k): v for k, v in self.label_map.items() if k != WILDCARD}

    def resolve(self, raw_label: str, tax: Taxonomy) -> Resolved | None:
        """Map a raw source label to the unified space. ``None`` means 'drop on purpose'."""
        key = norm_label(raw_label)
        if key in self._norm_map:
            target = self._norm_map[key]
        elif WILDCARD in self.label_map:
            target = self.label_map[WILDCARD]
        else:
            raise UnmappedLabelError(f"[{self.id}] label '{raw_label}' is not in label_map")

        if target is None:
            return None
        if target == AS_NON_PLANT:
            return Resolved("non_plant", None, f"{self.id}:{raw_label}")
        if target == KEEP:
            if self.role not in ("other_plant", "non_plant"):
                raise ValueError(f"[{self.id}] '{KEEP}' is only valid for other_plant/non_plant sources")
            return Resolved(self.role, None, f"{self.id}:{raw_label}")
        # A disease class: its status in the taxonomy decides known vs unknown_eval.
        cls = tax.get(self.crop, target)  # type: ignore[arg-type]
        return Resolved(cls.status, self.crop, cls.id)


@dataclass
class Registry:
    defaults: dict[str, Any]
    sources: list[Source]

    def get(self, source_id: str) -> Source:
        for s in self.sources:
            if s.id == source_id:
                return s
        raise KeyError(source_id)

    def enabled(self, ids: list[str] | None = None) -> list[Source]:
        if ids:
            return [self.get(i) for i in ids]  # explicit selection overrides `enabled`
        return [s for s in self.sources if s.enabled]


def load_registry(tax: Taxonomy, path: str | Path | None = None) -> Registry:
    path = Path(path) if path else CONFIG_DIR / "datasets.yaml"
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    seen: set[str] = set()
    sources: list[Source] = []
    for s in raw["sources"]:
        src = Source(
            id=s["id"], crop=s.get("crop"), role=s["role"], fetch=dict(s["fetch"]),
            license=str(s.get("license", "unknown")), citation=str(s.get("citation", "")),
            use_in_paper=bool(s.get("use_in_paper", False)), label_map=dict(s.get("label_map") or {}),
            enabled=bool(s.get("enabled", True)), train_only=bool(s.get("train_only", False)),
            max_per_class=s.get("max_per_class"),
            origin=dict(s.get("origin") or {}), notes=str(s.get("notes", "")),
        )
        _validate(src, tax)
        if src.id in seen:
            raise ValueError(f"duplicate source id '{src.id}'")
        seen.add(src.id)
        sources.append(src)
    return Registry(defaults=dict(raw.get("defaults") or {}), sources=sources)


def _validate(src: Source, tax: Taxonomy) -> None:
    where = f"[{src.id}]"
    if src.fetch.get("method") not in FETCH_METHODS:
        raise ValueError(f"{where} fetch.method must be one of {sorted(FETCH_METHODS)}")
    if src.fetch.get("method") == "hf" and str(src.fetch.get("config", "")).lower() == "augmented":
        raise ValueError(f"{where} augmented releases are not allowed (near-duplicate leakage)")
    if src.role == "known":
        if src.crop not in tax.crops:
            raise ValueError(f"{where} role 'known' needs a crop in {tax.crop_names}")
        for raw_label, target in src.label_map.items():
            if target is None:
                continue
            if target in (KEEP, AS_NON_PLANT):
                raise ValueError(f"{where} '{target}' not allowed in a disease source ({raw_label})")
            if not tax.has(src.crop, target):
                raise ValueError(f"{where} '{raw_label}' -> '{target}' is not a {src.crop} class")
    elif src.role in ("other_plant", "non_plant"):
        if src.crop is not None:
            raise ValueError(f"{where} {src.role} sources must have crop: null")
        for raw_label, target in src.label_map.items():
            if target not in (None, KEEP, AS_NON_PLANT):
                raise ValueError(f"{where} {src.role} label_map values must be null/@keep/@non_plant")
    else:
        raise ValueError(f"{where} role must be known|other_plant|non_plant (unknown_eval comes from taxonomy)")
