"""Iterate (PIL image, raw label) pairs from each kind of source.

Every fetcher yields ``(image, raw_label, origin_ref)`` where ``origin_ref`` lets a sample be
traced back to the exact item in the upstream dataset (HF row index, relative file path, COCO id).
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
import zipfile
from collections.abc import Iterator
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from govi_ai.data.registry import Source, norm_label

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
_TRAILING_INT = re.compile(r"(\d+)$")
Sample = tuple[Image.Image, str, str]

# COCO categories that show plants, vegetables or fruit -> such images are NOT "non_plant".
COCO_PLANT_CATEGORIES = {"potted plant", "broccoli", "carrot", "banana", "apple", "orange", "vase"}
COCO_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def iter_source(src: Source, data_root: Path, raw_root: Path | None = None) -> Iterator[Sample]:
    raw_root = raw_root or data_root / "raw"
    method = src.fetch["method"]
    if method == "hf":
        yield from _iter_hf(src)
    elif method in ("manual", "local"):
        root = raw_root / src.id if method == "manual" else _resolve(data_root, src.fetch["path"])
        if not root.exists():
            hint = f"download {src.fetch.get('url')} and unzip into {root}" if method == "manual" else \
                f"copy the folder to {root}"
            raise FileNotFoundError(f"[{src.id}] {root} not found: {hint}")
        yield from _iter_folder(src, root)
    elif method == "coco":
        yield from _iter_coco(src, raw_root / src.id)
    else:  # pragma: no cover - guarded by registry validation
        raise ValueError(method)


def _resolve(data_root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else data_root / path


# ---------------------------------------------------------------- Hugging Face
def _iter_hf(src: Source) -> Iterator[Sample]:
    from datasets import get_dataset_config_names, load_dataset, load_dataset_builder

    repo = src.fetch["repo"]
    config = src.fetch.get("config")
    if config is None:
        names = get_dataset_config_names(repo)
        if "raw" in names:
            config = "raw"
        elif names and set(names) <= {"augmented"}:
            raise ValueError(f"[{src.id}] {repo} only offers an augmented release")
        elif len(names) > 1 and "default" in names:
            config = "default"
        elif len(names) > 1:
            raise ValueError(f"[{src.id}] {repo} has configs {names}; set fetch.config explicitly")

    features = load_dataset_builder(repo, config).info.features
    image_col = next(k for k, v in features.items() if type(v).__name__ == "Image")
    label_col = next(k for k, v in features.items() if type(v).__name__ == "ClassLabel")
    names = features[label_col].names

    ds = load_dataset(repo, config, split=src.fetch.get("split", "train"), streaming=True)
    for i, row in enumerate(ds):
        img = row[image_col]
        if img is None:
            continue
        yield img, names[row[label_col]], f"hf:{repo}:{config or 'default'}:{i}"


# ---------------------------------------------------------------- folders (manual / local)
def _iter_folder(src: Source, root: Path) -> Iterator[Sample]:
    """Label = nearest ancestor folder whose name is in label_map (handles train/val/test or
    'diseased/<class>' nesting); falls back to the immediate parent folder."""
    known = {norm_label(k) for k in src.label_map}
    max_index = src.fetch.get("max_file_index")
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTS or not path.is_file():
            continue
        if max_index is not None:
            m = _TRAILING_INT.search(path.stem)
            if m is None or int(m.group(1)) > int(max_index):
                continue
        rel = path.relative_to(root)
        label = next((p for p in reversed(rel.parts[:-1]) if norm_label(p) in known), None)
        if label is None:
            label = rel.parts[-2] if len(rel.parts) > 1 else ""
        try:
            img = Image.open(path)
            img.load()
        except (UnidentifiedImageError, OSError) as e:
            log.warning("[%s] unreadable image %s: %s", src.id, rel, e)
            continue
        yield img, label, f"file:{rel.as_posix()}"


# ---------------------------------------------------------------- COCO (non-plant negatives)
def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed, well-known https/http host
    tmp.rename(dest)
    return dest


def _iter_coco(src: Source, root: Path) -> Iterator[Sample]:
    img_zip = _download(COCO_IMAGES_URL, root / "val2017.zip")
    ann_zip = _download(COCO_ANN_URL, root / "annotations_trainval2017.zip")
    with zipfile.ZipFile(ann_zip) as z:
        ann = json.loads(z.read("annotations/instances_val2017.json"))
    plant_cats = {c["id"] for c in ann["categories"] if c["name"] in COCO_PLANT_CATEGORIES}
    excluded = {a["image_id"] for a in ann["annotations"] if a["category_id"] in plant_cats}
    image_ids = sorted(im["id"] for im in ann["images"] if im["id"] not in excluded)
    image_ids = image_ids[: int(src.fetch.get("max_images", 4000))]
    with zipfile.ZipFile(img_zip) as z:
        for iid in image_ids:
            name = f"val2017/{iid:012d}.jpg"
            with z.open(name) as f:
                img = Image.open(f)
                img.load()
            yield img, "coco", f"coco:val2017:{iid}"
