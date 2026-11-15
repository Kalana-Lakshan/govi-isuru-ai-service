"""Pack a built dataset into one zip per source (plus a metadata zip) for Google Drive.

Thousands of small files sync to Drive slowly and read slowly from Colab; a few large zips
copy in seconds. Colab then unpacks them onto its local disk:

    python scripts/pack_dataset.py --data-root D:/govi-ai-data --out "G:/My Drive/govi-ai/dataset"
    # in Colab:
    python scripts/pack_dataset.py --unpack --out /content/drive/MyDrive/govi-ai/dataset --data-root /content/data

Images are already JPEG, so zips are stored uncompressed (ZIP_STORED): fast, nothing to gain.
Re-running only rewrites shards whose source manifest is newer than the shard.
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

META_FILES = ("manifest.csv", "build_report.json")


def pack(data_root: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    # Only images that survived deduplication (rows of the final manifest) are packed.
    with (data_root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        final: dict[str, list[str]] = {}
        for r in csv.DictReader(f):
            final.setdefault(r["source"], []).append(r["path"])
    index = {}
    for man in sorted((data_root / "manifests").glob("*.csv")):
        source = man.stem
        shard = out / f"images_{source}.zip"
        paths = final.get(source, [])
        if not paths:
            continue
        index[source] = {"shard": shard.name, "images": len(paths)}
        if shard.exists() and shard.stat().st_mtime >= man.stat().st_mtime:
            print(f"[skip] {shard.name} is up to date")
            continue
        tmp = shard.with_suffix(".zip.part")
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as z:
            for p in paths:
                z.write(data_root / p, arcname=p)
            z.write(man, arcname=f"manifests/{man.name}")
        tmp.replace(shard)
        print(f"[ok]   {shard.name}: {len(paths)} images, {shard.stat().st_size / 1e6:.0f} MB")

    meta = out / "metadata.zip"
    with zipfile.ZipFile(meta, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in META_FILES:
            if (data_root / name).exists():
                z.write(data_root / name, arcname=name)
        z.writestr("shards.json", json.dumps(index, indent=2))
    print(f"[ok]   {meta.name}")


def unpack(out: Path, data_root: Path) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    for shard in sorted(out.glob("*.zip")):
        with zipfile.ZipFile(shard) as z:
            z.extractall(data_root)
        print(f"unpacked {shard.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="folder holding the zip shards")
    ap.add_argument("--unpack", action="store_true")
    args = ap.parse_args()
    unpack(args.out, args.data_root) if args.unpack else pack(args.data_root, args.out)


if __name__ == "__main__":
    main()
