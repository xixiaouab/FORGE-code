from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BM25ThresholdPolicy:
    low: float
    high: float

    def select(self, scores):
        values = np.asarray(scores, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("BM25 scores must be finite")
        return np.where(values < self.low, 0, np.where(values < self.high, 1, 2))

    @classmethod
    def fit(cls, scores, utilities, *, split, grid_size=21):
        if split != "dev":
            raise ValueError("Threshold selection requires development data")
        scores, utilities = np.asarray(scores, float), np.asarray(utilities, float)
        if scores.ndim != 1 or utilities.shape != (len(scores), 3) or len(scores) == 0:
            raise ValueError("Expected scores (N,) and NoThink utility (N,3)")
        if grid_size < 2 or not np.isfinite(scores).all() or not np.isfinite(utilities).all():
            raise ValueError("Finite data and grid_size >= 2 required")
        grid = np.unique(np.r_[-np.inf, np.quantile(scores, np.linspace(0, 1, grid_size)), np.inf])
        best, best_value = None, -np.inf
        for i, low in enumerate(grid):
            for high in grid[i:]:
                candidate = cls(float(low), float(high))
                value = utilities[np.arange(len(scores)), candidate.select(scores)].mean()
                if value > best_value:
                    best, best_value = candidate, value
        return best


def dev_fallback(router_utilities, fixed_utilities, *, split, seed=42, n_resamples=10000):
    if split != "dev":
        raise ValueError("Fallback selection requires development data")
    routed, fixed = np.asarray(router_utilities, float), np.asarray(fixed_utilities, float)
    if routed.ndim != 1 or len(routed) == 0 or fixed.ndim != 2 or fixed.shape[0] != len(routed):
        raise ValueError("Expected routed utility (N,) and fixed utilities (N,K)")
    if n_resamples < 2 or not np.isfinite(routed).all() or not np.isfinite(fixed).all():
        raise ValueError("Finite utilities and n_resamples >= 2 required")
    best_fixed = int(fixed.mean(0).argmax())
    differences = routed - fixed[:, best_fixed]
    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples)
    for start in range(0, n_resamples, 256):
        end = min(n_resamples, start + 256)
        indices = rng.integers(0, len(routed), size=(end - start, len(routed)))
        values[start:end] = differences[indices].mean(1)
    low, high = np.quantile(values, [.025, .975])
    return {"use_router": bool(low > 0), "fixed_action": best_fixed,
            "dev_mean_utility_difference": float(differences.mean()),
            "confidence_interval": [float(low), float(high)], "seed": seed, "n_resamples": n_resamples}


def utility_oracle(utilities):
    values = np.asarray(utilities, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected a finite query-by-action utility matrix")
    return values.argmax(1)
