"""Build the harmonised Govi Isuru dataset.

Examples
--------
    # everything enabled in configs/datasets.yaml
    python scripts/build_dataset.py --data-root /content/drive/MyDrive/govi-ai/data

    # just two sources, then (re)finalize
    python scripts/build_dataset.py --data-root data --sources paddy_doctor rice_rlbf
    python scripts/build_dataset.py --data-root data --finalize-only

    # list sources and their status
    python scripts/build_dataset.py --list

Steps: fetch -> label harmonisation -> normalise (RGB JPEG, max side) -> per-source manifest
       -> finalize: cross-source near-duplicate grouping, label-conflict removal,
          in-domain split, leave-one-dataset-out columns, build_report.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from govi_ai.data.build import assign_groups, build_all, load_rows, write_manifest, write_report  # noqa: E402
from govi_ai.data.registry import load_registry  # noqa: E402
from govi_ai.data.split import add_lodo_columns, in_domain_split  # noqa: E402
from govi_ai.taxonomy import load_taxonomy  # noqa: E402


def finalize(reg, tax, data_root: Path, max_distance: int, seed: int, build_stats: list[dict]) -> dict:
    ids = [s.id for s in reg.sources]
    rows = load_rows(data_root, ids)
    if not rows:
        raise SystemExit("no per-source manifests found; build some sources first")
    dedupe = assign_groups(rows, max_distance=max_distance)
    in_domain_split(rows, seed=seed)
    paper_sources = [s.id for s in reg.sources if s.use_in_paper and s.role == "known"]
    lodo_cols = add_lodo_columns(rows, paper_sources, seed=seed)
    write_manifest(data_root / "manifest.csv", rows)

    known_counts: dict[str, dict[str, int]] = {}
    for crop in tax.crop_names:
        c = Counter(r["label"] for r in rows if r["crop"] == crop and r["role"] == "known")
        known_counts[crop] = {k: c.get(k, 0) for k in tax.known_classes(crop)}
    empty = {crop: [k for k, v in cnt.items() if v == 0] for crop, cnt in known_counts.items()}
    report = {
        "taxonomy_version": tax.version,
        "dedupe": dedupe,
        "rows": len(rows),
        "by_role": Counter(r["role"] for r in rows),
        "by_split": Counter(r["split"] for r in rows),
        "by_source": Counter(r["source"] for r in rows),
        "known_class_counts": known_counts,
        "known_classes_without_images": {k: v for k, v in empty.items() if v},
        "unknown_eval_counts": Counter(f"{r['crop']}/{r['label']}" for r in rows if r["role"] == "unknown_eval"),
        "lodo_columns": lodo_cols,
        "build": build_stats,
    }
    write_report(data_root / "build_report.json", report)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--raw-root", type=Path, help="manual downloads / archives (default: <data-root>/raw)")
    ap.add_argument("--sources", nargs="*", help="source ids to build (default: all enabled)")
    ap.add_argument("--force", action="store_true", help="rebuild sources that already have a manifest")
    ap.add_argument("--limit", type=int, help="max images per source (smoke tests)")
    ap.add_argument("--finalize-only", action="store_true")
    ap.add_argument("--max-distance", type=int, default=6, help="pHash Hamming distance for duplicates")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "datasets", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    tax = load_taxonomy()
    reg = load_registry(tax)
    if args.list:
        for s in reg.sources:
            flag = "on " if s.enabled else "off"
            print(f"[{flag}] {s.id:28s} {s.role:12s} {str(s.crop):6s} {s.fetch['method']:6s} paper={s.use_in_paper}"
                  f"  licence={s.license}")
        return

    args.data_root.mkdir(parents=True, exist_ok=True)
    stats: list[dict] = []
    if not args.finalize_only:
        stats = build_all(reg, tax, args.data_root, ids=args.sources, force=args.force, limit=args.limit,
                          raw_root=args.raw_root)
    report = finalize(reg, tax, args.data_root, args.max_distance, args.seed, stats)
    print(f"manifest: {args.data_root / 'manifest.csv'}  rows={report['rows']}")
    print(f"dedupe:   {report['dedupe']['duplicate_images']} duplicates, "
          f"{report['dedupe']['conflict_images_dropped']} label-conflict images dropped")
    if report["known_classes_without_images"]:
        print(f"WARNING known classes without images: {report['known_classes_without_images']}")


if __name__ == "__main__":
    main()
