"""Materialise sources into one harmonised, deduplicated image set.

Layout under ``data_root``::

    raw/<source>/...                 manual downloads (you put them here)
    images/<source>/<ab>/<sha1>.jpg  normalised images (RGB JPEG, longest side <= max_side)
    manifests/<source>.csv           per-source rows (lets you rebuild one source at a time)
    manifest.csv                     all sources + dedupe groups + splits  (written by finalize)
    build_report.json                counts, dropped labels, duplicate/conflict statistics
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from govi_ai.data.dedupe import dihedral_hashes, group_near_duplicates, hex_hashes, parse_hex_hashes
from govi_ai.data.fetch import iter_source
from govi_ai.data.registry import Registry, Source
from govi_ai.taxonomy import Taxonomy

log = logging.getLogger(__name__)

MANIFEST_FIELDS = [
    "sample_id", "path", "source", "source_label", "role", "crop", "label",
    "width", "height", "use_in_paper", "train_only", "origin_ref", "hashes",
]


@dataclass
class Row:
    sample_id: str
    path: str
    source: str
    source_label: str
    role: str
    crop: str
    label: str
    width: int
    height: int
    use_in_paper: bool
    train_only: bool
    origin_ref: str
    hashes: str


def normalise_image(img: Image.Image, max_side: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
    return img


def build_source(
    src: Source, tax: Taxonomy, data_root: Path, *, max_side: int = 512, jpeg_quality: int = 90,
    limit: int | None = None, raw_root: Path | None = None,
) -> dict:
    """Fetch one source, write normalised images and ``manifests/<id>.csv``. Returns stats.

    ``raw_root`` (default ``data_root/raw``) holds manual downloads and archives such as COCO;
    keep it on local disk when ``data_root`` is a synced folder.
    """
    out_dir = data_root / "images" / src.id
    rows: list[Row] = []
    per_class: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    seen_ids: set[str] = set()

    for i, (img, raw_label, ref) in enumerate(iter_source(src, data_root, raw_root)):
        if limit is not None and i >= limit:
            break
        res = src.resolve(raw_label, tax)
        if res is None:
            dropped[raw_label] += 1
            continue
        if src.max_per_class and per_class[res.label] >= src.max_per_class:
            continue
        norm = normalise_image(img, max_side)
        buf = io.BytesIO()
        norm.save(buf, format="JPEG", quality=jpeg_quality)
        data = buf.getvalue()
        sid = hashlib.sha1(data).hexdigest()
        if sid in seen_ids:  # byte-identical within the source
            continue
        seen_ids.add(sid)
        rel = Path("images") / src.id / sid[:2] / f"{sid}.jpg"
        dest = data_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(data)
        per_class[res.label] += 1
        rows.append(Row(
            sample_id=sid, path=rel.as_posix(), source=src.id, source_label=raw_label, role=res.role,
            crop=res.crop or "", label=res.label, width=norm.width, height=norm.height,
            use_in_paper=src.use_in_paper, train_only=src.train_only, origin_ref=ref, hashes=hex_hashes(dihedral_hashes(norm)),
        ))

    man_dir = data_root / "manifests"
    man_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(man_dir / f"{src.id}.csv", rows)
    stats = {"source": src.id, "kept": len(rows), "per_class": dict(per_class), "dropped": dict(dropped)}
    log.info("[%s] kept %d images (%d dropped labels)", src.id, len(rows), sum(dropped.values()))
    return stats


def build_all(reg: Registry, tax: Taxonomy, data_root: Path, *, ids: list[str] | None = None,
              force: bool = False, limit: int | None = None, raw_root: Path | None = None) -> list[dict]:
    max_side = int(reg.defaults.get("max_side", 512))
    quality = int(reg.defaults.get("jpeg_quality", 90))
    stats = []
    for src in reg.enabled(ids):
        if not force and (data_root / "manifests" / f"{src.id}.csv").exists():
            log.info("[%s] already built (use --force to rebuild)", src.id)
            continue
        try:
            stats.append(build_source(src, tax, data_root, max_side=max_side, jpeg_quality=quality,
                                      limit=limit, raw_root=raw_root))
        except FileNotFoundError as e:  # manual/local source not downloaded yet: report, continue
            log.warning("%s", e)
            stats.append({"source": src.id, "skipped": str(e)})
        except Exception as e:  # network drop, sleep/resume, bad archive: keep going; rerun resumes this source
            log.error("[%s] failed: %r (rerun the build to retry; finished sources are skipped)", src.id, e)
            stats.append({"source": src.id, "failed": repr(e)})
    return stats


# ---------------------------------------------------------------- finalize: dedupe across sources
def load_rows(data_root: Path, source_ids: list[str]) -> list[dict]:
    rows: list[dict] = []
    for sid in source_ids:
        p = data_root / "manifests" / f"{sid}.csv"
        if p.exists():
            with p.open(newline="", encoding="utf-8") as f:
                rows.extend(csv.DictReader(f))
    return rows


def assign_groups(rows: list[dict], max_distance: int = 6, one_per_group: bool = True) -> dict:
    """Add ``group_id`` to every row and drop clusters whose members disagree on the label.

    A cluster that is labelled e.g. brown_spot in one source and blast in another is label noise;
    keeping either copy would train or test on a wrong label, so the whole cluster is dropped.
    With ``one_per_group`` only the highest-resolution member of each cluster is kept, which
    removes pre-augmented copies (they add no information and would inflate test sets).
    Returns statistics for the build report.
    """
    groups = group_near_duplicates([parse_hex_hashes(r["hashes"]) for r in rows], max_distance)
    members: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)

    conflicts: Counter[str] = Counter()
    keep = [True] * len(rows)
    cross_source = 0
    for g, idx in members.items():
        labels = {(rows[i]["role"], rows[i]["crop"], rows[i]["label"]) for i in idx}
        if len({rows[i]["source"] for i in idx}) > 1:
            cross_source += 1
        # Negatives may legitimately share pictures across sources; only disease labels conflict.
        disease_labels = {lab for lab in labels if lab[0] in ("known", "unknown_eval")}
        if len(disease_labels) > 1 or (disease_labels and len(labels) > 1):
            conflicts[" | ".join(sorted("/".join(x for x in lab if x) for lab in labels))] += len(idx)
            for i in idx:
                keep[i] = False
        elif one_per_group and len(idx) > 1:
            best = max(idx, key=lambda i: (int(rows[i]["width"]) * int(rows[i]["height"]), -i))
            for i in idx:
                keep[i] = i == best
    for i, r in enumerate(rows):
        r["group_id"] = f"g{groups[i]:07d}"
    n_groups = len(members)
    rows[:] = [r for i, r in enumerate(rows) if keep[i]]
    return {
        "images_in": len(keep), "images_kept": len(rows), "groups": n_groups,
        "duplicate_images": len(keep) - n_groups, "cross_source_groups": cross_source,
        "conflict_images_dropped": int(sum(conflicts.values())), "conflicts": dict(conflicts.most_common(30)),
    }


def _write_csv(path: Path, rows: list) -> None:
    rows = [asdict(r) if isinstance(r, Row) else r for r in rows]
    fields = list(rows[0].keys()) if rows else MANIFEST_FIELDS
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_manifest(path: Path, rows: list[dict]) -> None:
    _write_csv(path, rows)


def write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
