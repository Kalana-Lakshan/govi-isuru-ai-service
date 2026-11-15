"""Open-set ("is this a known disease?") scores. Convention: higher = more likely known.

Logit-based:   MSP (max softmax probability), max-logit, energy
Feature-based: Mahalanobis distance (class means + shared covariance), deep kNN cosine similarity

Feature-based scores are the basis of the prototype library: an officer-verified new disease
becomes a new class mean without retraining the network.
"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def msp(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return softmax(logits, temperature).max(1)


def max_logit(logits: np.ndarray) -> np.ndarray:
    return logits.max(1)


def energy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Negative free energy, T·logsumexp(logits/T) (Liu et al. 2020)."""
    z = logits / temperature
    m = z.max(1, keepdims=True)
    return temperature * (m[:, 0] + np.log(np.exp(z - m).sum(1)))


class Mahalanobis:
    """Class-conditional Gaussians with a shared covariance (Lee et al. 2018)."""

    def fit(self, feats: np.ndarray, labels: np.ndarray, shrinkage: float = 1e-3) -> "Mahalanobis":
        self.classes_ = np.unique(labels)
        self.means_ = np.stack([feats[labels == c].mean(0) for c in self.classes_])
        centered = feats - self.means_[np.searchsorted(self.classes_, labels)]
        cov = centered.T @ centered / max(1, len(feats) - 1)
        cov += shrinkage * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
        self.precision_ = np.linalg.inv(cov)
        return self

    def distances(self, feats: np.ndarray) -> np.ndarray:
        # (x-m)ᵀP(x-m) = xᵀPx - 2xᵀPm + mᵀPm, without materialising n×k×d differences.
        xp = feats @ self.precision_
        mp = self.means_ @ self.precision_
        return ((xp * feats).sum(1)[:, None] - 2 * xp @ self.means_.T
                + (mp * self.means_).sum(1)[None, :])

    def add_class(self, label, feats: np.ndarray) -> None:
        """Prototype library: register a new (officer-verified) class without refitting."""
        self.classes_ = np.append(self.classes_, label)
        self.means_ = np.vstack([self.means_, feats.mean(0, keepdims=True)])

    def score(self, feats: np.ndarray, chunk: int = 2048) -> np.ndarray:
        return np.concatenate([-self.distances(feats[i:i + chunk]).min(1) for i in range(0, len(feats), chunk)])


def knn_score(train_feats: np.ndarray, feats: np.ndarray, k: int = 10, chunk: int = 1024) -> np.ndarray:
    """Deep kNN (Sun et al. 2022): cosine similarity to the k-th nearest training feature."""
    tr = train_feats / (np.linalg.norm(train_feats, axis=1, keepdims=True) + 1e-8)
    out = []
    for i in range(0, len(feats), chunk):
        q = feats[i:i + chunk]
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
        sims = q @ tr.T
        kk = min(k, sims.shape[1])
        out.append(-np.partition(-sims, kk - 1, axis=1)[:, kk - 1])
    return np.concatenate(out)


def fit_temperature(logits: np.ndarray, y: np.ndarray, grid=np.logspace(-1.5, 1.5, 61)) -> float:
    """Temperature scaling (Guo et al. 2017) by grid search on validation NLL."""
    best_t, best_nll = 1.0, np.inf
    idx = np.arange(len(y))
    for t in grid:
        p = softmax(logits, t)[idx, y]
        nll = -np.log(np.clip(p, 1e-12, 1)).mean()
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t
