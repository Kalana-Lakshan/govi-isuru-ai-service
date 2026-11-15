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
  against ``max_distance``. Clusters = connected components (union-find).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

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


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _blocks(h: int) -> Iterable[tuple[int, int]]:
    mask = (1 << _BLOCK_BITS) - 1
    for b in range(_N_BLOCKS):
        yield b, (h >> (b * _BLOCK_BITS)) & mask


def group_near_duplicates(
    variant_hashes: Sequence[Sequence[int]], max_distance: int = 6
) -> list[int]:
    """Cluster images whose hashes are within ``max_distance`` under any dihedral transform.

    Args:
        variant_hashes: per image, its 8 dihedral hashes (index 0 = identity).
        max_distance: max Hamming distance (of 64 bits) to call two images duplicates.

    Returns:
        group id per image (the smallest member index of its cluster).
    """
    n = len(variant_hashes)
    index: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, hs in enumerate(variant_hashes):
        for key in _blocks(hs[0]):
            index[key].append(i)

    uf = _UnionFind(n)
    for i, hs in enumerate(variant_hashes):
        seen: set[int] = set()
        for h in hs:
            for key in _blocks(h):
                for j in index.get(key, ()):
                    if j == i or j in seen:
                        continue
                    seen.add(j)
                    if hamming(h, variant_hashes[j][0]) <= max_distance:
                        uf.union(i, j)
    return [uf.find(i) for i in range(n)]


def hex_hashes(hs: Sequence[int]) -> str:
    """Compact text form for the manifest: 8 hex hashes joined by '.'."""
    return ".".join(f"{h:016x}" for h in hs)


def parse_hex_hashes(s: str) -> list[int]:
    return [int(x, 16) for x in s.split(".")]
