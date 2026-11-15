import numpy as np
from PIL import Image

from govi_ai.data.dedupe import dihedral_hashes, group_near_duplicates, hamming, hex_hashes, parse_hex_hashes, phash


def _img(seed: int, size=(200, 150)) -> Image.Image:
    rng = np.random.default_rng(seed)
    # smooth random blobs: closer to photo statistics than white noise
    base = rng.random((12, 16, 3))
    return Image.fromarray((base * 255).astype(np.uint8)).resize(size, Image.Resampling.BICUBIC)


def test_identity_variant_equals_phash():
    im = _img(0)
    assert dihedral_hashes(im)[0] == phash(im)


def test_hex_roundtrip():
    hs = dihedral_hashes(_img(1))
    assert parse_hex_hashes(hex_hashes(hs)) == hs


def test_recompressed_copy_is_close():
    import io
    im = _img(2)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=60)
    copy = Image.open(io.BytesIO(buf.getvalue()))
    assert hamming(phash(im), phash(copy)) <= 6


def test_groups_flips_rotations_and_resizes_but_not_different_images():
    a = _img(3)
    variants = [
        a,
        a.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        a.transpose(Image.Transpose.ROTATE_90),
        a.transpose(Image.Transpose.ROTATE_180),
        a.resize((100, 75)),
    ]
    others = [_img(s) for s in range(10, 20)]
    groups = group_near_duplicates([dihedral_hashes(im) for im in variants + others], max_distance=6)
    assert len(set(groups[: len(variants)])) == 1, "augmented copies must share a group"
    assert len(set(groups[len(variants):])) == len(others), "distinct images must not merge"
    assert groups[0] not in groups[len(variants):]
