"""Falsification: actively probe the current best.

The loop occasionally swaps an exploit step for one that is *trying to be
wrong* — generate small perturbations of the leading candidate and submit the
one the surrogate is most confident will fail. If reality still rewards it,
the surrogate's local gradient is mistaken and we just gathered the most
informative possible counter-example.
"""

from __future__ import annotations

import numpy as np

from .candidates import Candidate, featurize, perturb
from .neural import GPSurrogate
from .symbolic import symbolic_check


def falsify_neighbors(
    best: Candidate,
    model: GPSurrogate,
    rng: np.random.Generator,
    n: int = 32,
    scale: float = 0.05,
) -> Candidate | None:
    """Generate `n` small perturbations of `best`, return the one the model
    predicts will be *worst*. Returns None if no perturbation passes the
    symbolic veto.
    """
    probes: list[Candidate] = []
    for _ in range(n):
        p = perturb(best, rng, scale=scale)
        if symbolic_check(p).ok:
            probes.append(p)
    if not probes:
        return None
    feats = np.stack([featurize(p) for p in probes])
    mean, _ = model.predict(feats)
    worst = int(np.argmin(mean))
    return probes[worst]
