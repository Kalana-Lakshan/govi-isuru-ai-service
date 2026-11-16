"""Group stored UNKNOWN_DISEASE cases into candidate new diseases for officer review (numpy only).

Density-seeded centroid clustering on cosine similarity (no transitive chaining):
  1. every case's density = number of other cases with cosine similarity >= ``threshold``;
  2. the densest unassigned case seeds a cluster; every unassigned case within ``threshold`` of the
     seed joins; the centroid is recomputed and membership re-checked once against it;
  3. repeat; clusters smaller than ``min_size`` become noise (reviewed individually).

Deterministic, O(n^2) memory, intended for up to a few thousand cases per crop. The threshold is
calibrated offline on real held-out disorders (scripts/officer_loop_experiment.py).
"""

from __future__ import annotations

import numpy as np


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def cluster_cases(emb: np.ndarray, threshold: float = 0.8, min_size: int = 3) -> tuple[list[np.ndarray], np.ndarray]:
    """Returns (clusters as arrays of row indices, noise row indices)."""
    n = len(emb)
    if n == 0:
        return [], np.zeros(0, dtype=int)
    z = _normalise(emb)
    sim = z @ z.T
    density = (sim >= threshold).sum(1) - 1
    unassigned = np.ones(n, dtype=bool)
    clusters: list[np.ndarray] = []
    noise: list[int] = []
    for seed in np.argsort(-density, kind="stable"):
        if not unassigned[seed]:
            continue
        members = np.flatnonzero(unassigned & (sim[seed] >= threshold))
        centroid = _normalise(z[members].mean(0, keepdims=True))[0]
        members = np.flatnonzero(unassigned & (z @ centroid >= threshold))
        if seed not in members:
            members = np.append(members, seed)
        unassigned[members] = False
        if len(members) >= min_size:
            clusters.append(np.sort(members))
        else:
            noise.extend(members.tolist())
    return clusters, np.array(sorted(noise), dtype=int)


def describe(emb: np.ndarray, clusters: list[np.ndarray]) -> list[dict]:
    """Per cluster: size, cohesion (mean cosine to centroid) and the most central member (an
    officer looks at that photo first)."""
    z = _normalise(emb)
    out = []
    for c in clusters:
        centroid = _normalise(z[c].mean(0, keepdims=True))[0]
        cos = z[c] @ centroid
        out.append({"size": int(len(c)), "cohesion": round(float(cos.mean()), 4),
                    "representative": int(c[int(cos.argmax())])})
    return out
