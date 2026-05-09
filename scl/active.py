"""Active-learning acquisitions.

Three Bayesian-optimization variants over the surrogate, plus a random
baseline:

* **UCB** — ``mean + κ·std``. Cheapest, deterministic, and the workhorse the
  rest of the codebase has used by default.
* **EI** — Expected Improvement over the current best. Calibrated balance of
  exploit/explore; tends to converge faster on smooth landscapes.
* **Thompson** — Sample one realization per candidate from the marginal GP
  posterior and pick the argmax. Naturally diversifying.

All four return ``(chosen, mean, std, score)`` aligned by chosen-order so the
caller can log per-pick predictions uniformly.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .candidates import Candidate, featurize
from .neural import GPSurrogate


_VEC_ERF = np.vectorize(math.erf)
_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _phi_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _VEC_ERF(z / _SQRT_2))


def _phi_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / _SQRT_2PI


def ucb_select(
    candidates: Sequence[Candidate],
    model: GPSurrogate,
    kappa: float = 2.0,
    k: int = 1,
) -> tuple[list[Candidate], np.ndarray, np.ndarray, np.ndarray]:
    """Upper-confidence-bound: ``mean + κ·std``."""
    if not candidates:
        raise ValueError("no candidates to select from")
    feats = np.stack([featurize(c) for c in candidates])
    mean, std = model.predict(feats)
    ucb = mean + kappa * std
    order = np.argsort(-ucb)[:k]
    chosen = [candidates[int(i)] for i in order]
    return chosen, mean[order], std[order], ucb[order]


def ei_select(
    candidates: Sequence[Candidate],
    model: GPSurrogate,
    current_best: float,
    k: int = 1,
    xi: float = 0.0,
) -> tuple[list[Candidate], np.ndarray, np.ndarray, np.ndarray]:
    """Expected Improvement over ``current_best``.

    ``xi`` shifts the improvement threshold — positive ``xi`` makes the rule
    more exploratory by demanding bigger improvements.
    """
    if not candidates:
        raise ValueError("no candidates to select from")
    feats = np.stack([featurize(c) for c in candidates])
    mean, std = model.predict(feats)
    improvement = mean - current_best - xi
    safe_std = np.maximum(std, 1e-9)
    z = improvement / safe_std
    ei = improvement * _phi_cdf(z) + safe_std * _phi_pdf(z)
    ei = np.where(std > 1e-9, ei, 0.0)
    order = np.argsort(-ei)[:k]
    chosen = [candidates[int(i)] for i in order]
    return chosen, mean[order], std[order], ei[order]


def thompson_select(
    candidates: Sequence[Candidate],
    model: GPSurrogate,
    rng: np.random.Generator,
    k: int = 1,
) -> tuple[list[Candidate], np.ndarray, np.ndarray, np.ndarray]:
    """Marginal Thompson sampling: one draw per candidate, take the argmax."""
    if not candidates:
        raise ValueError("no candidates to select from")
    feats = np.stack([featurize(c) for c in candidates])
    mean, std = model.predict(feats)
    sample = mean + std * rng.standard_normal(len(mean))
    order = np.argsort(-sample)[:k]
    chosen = [candidates[int(i)] for i in order]
    return chosen, mean[order], std[order], sample[order]


def random_select(
    candidates: Sequence[Candidate],
    rng: np.random.Generator,
    k: int = 1,
) -> list[Candidate]:
    """Baseline: pick uniformly at random from the survivor pool."""
    if not candidates:
        raise ValueError("no candidates to select from")
    idx = rng.choice(len(candidates), size=min(k, len(candidates)), replace=False)
    return [candidates[int(i)] for i in idx]
