"""Near-duplicate detection with rotation/flip-invariant perceptual hashing.

Why: public plant-disease datasets are frequently re-packaged from each other and often ship
pre-augmented copies (flips, 90° rotations, re-encodes). If two copies of one photo land in
train and test, reported accuracy is inflated. Every near-duplicate cluster therefore becomes
one *group*, and splits are made at group level.

Method
------
* pHash (DCT of a 32x32 grey image, sign of the 8x8 low-frequency block vs. its median) -> 64 bit.
* Each image is indexed by its own hash; it is *queried* with the hashes of all 8 dihedral
  transforms (identity, rotations, flips), so augmented copies are found too.
* Candidate pairs come from multi-index hashing on 4 x 16-bit blocks (pigeonhole: any pair within
  Hamming distance <= 3 shares a block; most pairs up to ~8 are found as well), then are verified
  against ``max_distance`` and, in the build, by pixel correlation of aligned, background-subtracted
  thumbnails (``pixel_match``). Groups are formed by leader clustering, never by chaining.
* This stage is deliberately high-precision (exact copies, flips/rotations, re-encodes). Fuzzier
  near-duplicates (crops, zooms) are left to the embedding-based check of phase 2.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence

import numpy as np
from PIL import Image

HASH_SIZE = 8
_IMG_SIZE = HASH_SIZE * 4  # 32
_N_BLOCKS = 4
_BLOCK_BITS = 64 // _N_BLOCKS


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    m[0] *= 1 / np.sqrt(2)
    return m * np.sqrt(2 / n)


_DCT = _dct_matrix(_IMG_SIZE)
_BIT_WEIGHTS = (1 << np.arange(63, -1, -1, dtype=np.uint64)).astype(np.uint64)


def _grey32(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L").resize((_IMG_SIZE, _IMG_SIZE), Image.Resampling.LANCZOS),
                      dtype=np.float64)


def _phash_from_grey(g: np.ndarray) -> int:
    d = _DCT @ g @ _DCT.T
    low = d[:HASH_SIZE, :HASH_SIZE].flatten()
    bits = (low > np.median(low)).astype(np.uint64)
    return int((bits * _BIT_WEIGHTS).sum())


def phash(img: Image.Image) -> int:
    """64-bit perceptual hash of an image (as a Python int)."""
    return _phash_from_grey(_grey32(img))


def dihedral_hashes(img: Image.Image) -> list[int]:
    """pHashes of the 8 dihedral transforms; element 0 is the identity (== ``phash(img)``)."""
    g = _grey32(img)
    out = []
    for k in range(4):
        r = np.rot90(g, k)
        out.append(_phash_from_grey(r))
        out.append(_phash_from_grey(np.fliplr(r)))
    return out


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _blocks(h: int) -> Iterable[tuple[int, int]]:
    mask = (1 << _BLOCK_BITS) - 1
    for b in range(_N_BLOCKS):
        yield b, (h >> (b * _BLOCK_BITS)) & mask


def group_near_duplicates(
    variant_hashes: Sequence[Sequence[int]], max_distance: int = 6,
    verify: Callable[[int, int, int], bool] | None = None,
) -> list[int]:
    """Group duplicates by *leader* clustering (no transitive chaining).

    Each image is compared only with existing group leaders; it joins the closest leader whose hash is
    within ``max_distance`` under some dihedral transform *and* that passes ``verify(i, leader, variant)``,
    otherwise it starts a new group. Plain connected components (union-find) chained thousands of
    unrelated images together on the real data (one 19k-image "group" spanning rice, tea and COCO),
    because "single leaf on a white sheet" photos all have similar coarse hashes.

    Args:
        variant_hashes: per image, its 8 dihedral hashes (index 0 = identity).
        max_distance: max Hamming distance (of 64 bits) for a candidate pair.
        verify: optional pixel-level check; ``variant`` is the dihedral index that matched.

    Returns:
        group id per image (the index of its group's leader).
    """
    index: dict[tuple[int, int], list[int]] = defaultdict(list)  # leaders only
    group = [-1] * len(variant_hashes)
    for i, hs in enumerate(variant_hashes):
        cands: dict[int, tuple[int, int]] = {}
        for v, h in enumerate(hs):
            for key in _blocks(h):
                for leader in index.get(key, ()):
                    d = hamming(h, variant_hashes[leader][0])
                    if d <= max_distance and (leader not in cands or d < cands[leader][0]):
                        cands[leader] = (d, v)
        chosen = None
        for leader, (d, v) in sorted(cands.items(), key=lambda kv: kv[1][0]):
            if verify is None or verify(i, leader, v):
                chosen = leader
                break
        if chosen is None:
            group[i] = i
            for key in _blocks(hs[0]):
                index[key].append(i)
        else:
            group[i] = chosen
    return group


THUMB = 32


def thumbnail(img: Image.Image) -> np.ndarray:
    """Background-subtracted, unit-norm 32x32 RGB thumbnail (border median = background estimate),
    so that 'a leaf on a white sheet' photos are compared on the leaf, not on the sheet."""
    g = np.asarray(img.convert("RGB").resize((THUMB, THUMB), Image.Resampling.BILINEAR), dtype=np.float32)
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    g = g - np.median(border, 0)
    g = g - g.mean()
    return g / (np.linalg.norm(g) + 1e-6)


def dihedral(a: np.ndarray, v: int) -> np.ndarray:
    """Variant ``v`` in the same order as ``dihedral_hashes`` (rot90 k = v // 2, then flip if odd)."""
    a = np.rot90(a, v // 2)
    return np.fliplr(a) if v % 2 else a


def pixel_match(a: np.ndarray, b: np.ndarray, variant: int, min_corr: float = 0.98) -> bool:
    """``a`` (transformed by ``variant``) vs ``b``: correlation of background-subtracted thumbnails.
    0.98 was chosen on 3,000 real candidate pairs: it keeps ~2 % of false pHash matches (different
    leaves on white sheets) while keeping exact copies, flips/rotations and re-encodes."""
    return float((dihedral(a, variant) * b).sum()) >= min_corr


def hex_hashes(hs: Sequence[int]) -> str:
    """Compact text form for the manifest: 8 hex hashes joined by '.'."""
    return ".".join(f"{h:016x}" for h in hs)


def parse_hex_hashes(s: str) -> list[int]:
    return [int(x, 16) for x in s.split(".")]
