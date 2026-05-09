"""Active-learning acquisition.

UCB (mean + kappa * std) with kappa=0 collapsing to pure exploitation, and
larger kappa exploring uncertain regions. The selection is the only place where
'which experiment do we run next' lives.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .candidates import Candidate, featurize
from .neural import GPSurrogate


def ucb_select(
    candidates: Sequence[Candidate],
    model: GPSurrogate,
    kappa: float = 2.0,
    k: int = 1,
) -> tuple[list[Candidate], np.ndarray, np.ndarray, np.ndarray]:
    """Return (chosen, mean, std, ucb) — all aligned by chosen-order."""
    if not candidates:
        raise ValueError("no candidates to select from")
    feats = np.stack([featurize(c) for c in candidates])
    mean, std = model.predict(feats)
    ucb = mean + kappa * std
    order = np.argsort(-ucb)[:k]
    chosen = [candidates[int(i)] for i in order]
    return chosen, mean[order], std[order], ucb[order]


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
