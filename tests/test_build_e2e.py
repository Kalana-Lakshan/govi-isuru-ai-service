"""End-to-end build on a tiny on-disk 'local' source (test fixtures only, never training data)."""

import csv
import textwrap

import numpy as np
from PIL import Image

from govi_ai.data.build import assign_groups, build_source, load_rows
from govi_ai.data.registry import load_registry
from govi_ai.data.split import in_domain_split
from govi_ai.taxonomy import load_taxonomy


def _write_img(path, seed):
    rng = np.random.default_rng(seed)
    arr = (rng.random((10, 10, 3)) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).resize((900, 700), Image.Resampling.BICUBIC).save(path)


def test_build_local_source(tmp_path):
    root = tmp_path / "data"
    src_dir = root / "legacy" / "tiny"
    # nested train/test layout + a duplicate (flipped) + a dropped label folder
    for i in range(30):
        _write_img(src_dir / "train" / "Healthy" / f"h{i}.png", i)
        _write_img(src_dir / "test" / "Leaf Spot" / f"s{i}.png", 100 + i)
    (src_dir / "test" / "Healthy").mkdir(parents=True)
    Image.open(src_dir / "train" / "Healthy" / "h0.png").transpose(
        Image.Transpose.FLIP_LEFT_RIGHT).save(src_dir / "test" / "Healthy" / "h0_flip.png")
    _write_img(src_dir / "train" / "Junk" / "j.png", 999)

    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(textwrap.dedent("""
        defaults: {max_side: 256}
        sources:
          - id: tiny
            crop: chili
            role: known
            fetch: {method: local, path: legacy/tiny}
            license: test
            citation: test
            use_in_paper: false
            label_map: {Healthy: healthy, Leaf Spot: leaf_spot, Junk: null}
    """), encoding="utf-8")
    tax = load_taxonomy()
    reg = load_registry(tax, cfg)

    stats = build_source(reg.get("tiny"), tax, root, max_side=256)
    assert stats["kept"] == 61 and stats["dropped"] == {"Junk": 1}

    with (root / "manifests" / "tiny.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert max(int(r["width"]) for r in rows) <= 256
    assert {r["label"] for r in rows} == {"healthy", "leaf_spot"}
    assert all((root / r["path"]).exists() for r in rows)

    rows = load_rows(root, ["tiny"])
    rep = assign_groups(rows)
    assert rep["duplicate_images"] >= 1 and rep["images_kept"] == 60  # flipped copy collapsed
    in_domain_split(rows)
    assert {r["split"] for r in rows} <= {"train", "val", "test"}
