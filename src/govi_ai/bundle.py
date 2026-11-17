"""Assemble the non-model parts of a serving bundle: class metadata, yield and suitability files."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from govi_ai import CONFIG_DIR
from govi_ai.taxonomy import Taxonomy

YIELD_FILES = ("forecasts.json", "history.json", "backtest.json")
SUITABILITY_FILES = ("crops.json", "districts.json")


def class_catalog(tax: Taxonomy, info_path: Path | None = None) -> dict:
    """Per crop and class: type, status, display names and (agronomist-provided) advice.

    Advice comes from ``configs/disease_info.yaml``. Empty fields stay empty: the service never
    invents treatment text.
    """
    info_path = info_path or CONFIG_DIR / "disease_info.yaml"
    info = yaml.safe_load(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    info = (info or {}).get("crops", {})
    out: dict = {"taxonomy_version": tax.version, "gate_classes": tax.gate_classes, "crops": {}}
    for crop, classes in tax.crops.items():
        out["crops"][crop] = {}
        for cid, c in classes.items():
            extra = (info.get(crop) or {}).get(cid) or {}
            out["crops"][crop][cid] = {
                "type": c.type, "status": c.status,
                "name": {"en": c.en, "si": c.si or extra.get("si", ""), "ta": c.ta or extra.get("ta", "")},
                "advice": {lang: extra.get(f"advice_{lang}", "") for lang in ("en", "si", "ta")},
                "description": {"en": extra.get("description_en", "")},
                "severity": extra.get("severity") or None,
                "advice_source": extra.get("source", ""),
            }
    return out


def add_extras(bundle_dir: Path, tax: Taxonomy, yield_dir: Path | None = None,
               suitability_dir: Path | None = None) -> dict:
    """Copy yield / suitability files and write classes.json into ``bundle_dir``. Returns what was added."""
    bundle_dir = Path(bundle_dir)
    added: dict = {}
    (bundle_dir / "classes.json").write_text(json.dumps(class_catalog(tax), indent=1, ensure_ascii=False),
                                             encoding="utf-8")
    added["classes"] = True
    for name, src, files in (("yield", yield_dir, YIELD_FILES), ("suitability", suitability_dir, SUITABILITY_FILES)):
        if not src:
            continue
        src = Path(src)
        dest = bundle_dir / name
        dest.mkdir(parents=True, exist_ok=True)
        copied = [f for f in files if (src / f).exists()]
        for f in copied:
            shutil.copy2(src / f, dest / f)
        added[name] = copied
    return added
