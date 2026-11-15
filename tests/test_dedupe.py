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


def _leaf_on_white(seed: int, size=(200, 160)) -> Image.Image:
    """A synthetic dark 'leaf' ellipse on a white sheet: the case where coarse hashes collide."""
    from PIL import ImageDraw

    rng = np.random.default_rng(seed)
    im = Image.new("RGB", size, (245, 245, 240))
    d = ImageDraw.Draw(im)
    w, h = size
    cx, cy = w / 2 + rng.normal(0, 8), h / 2 + rng.normal(0, 8)
    rx, ry = w * rng.uniform(0.25, 0.4), h * rng.uniform(0.12, 0.25)
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(40, int(rng.uniform(80, 140)), 40))
    for _ in range(6):  # lesions
        x, y = cx + rng.normal(0, rx / 2), cy + rng.normal(0, ry / 2)
        d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(120, 90, 40))
    return im


def test_pixel_verification_rejects_different_leaves_but_keeps_copies():
    from govi_ai.data.dedupe import pixel_match, thumbnail

    a = _leaf_on_white(1)
    flipped = a.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    others = [_leaf_on_white(s) for s in range(2, 12)]
    imgs = [a, flipped, *others]
    thumbs = [thumbnail(im) for im in imgs]

    def verify(i, leader, v):
        return pixel_match(thumbs[i], thumbs[leader], v)

    hashes = [dihedral_hashes(im) for im in imgs]
    plain = group_near_duplicates(hashes, max_distance=10)
    checked = group_near_duplicates(hashes, max_distance=10, verify=verify)
    assert checked[0] == checked[1], "a flipped copy must still be grouped"
    assert len(set(checked[2:])) == len(others), "different leaves must not be grouped"
    assert len(set(plain)) <= len(set(checked))


def test_leader_clustering_does_not_chain():
    # Hashes 0 -> 3 bits -> 6 bits from the first: with max_distance 4, the third is within 4 of the
    # second but not of the leader, so it must form its own group (union-find would chain all three).
    h0 = 0
    h1 = 0b111
    h2 = 0b111111
    groups = group_near_duplicates([[h0] * 8, [h1] * 8, [h2] * 8], max_distance=4)
    assert groups[0] == groups[1] and groups[2] != groups[0]
