"""Information-manifold engine.

Each candidate's feature vector is a point in a 7-D manifold of materials. We
estimate the *curvature* of the surrogate's predicted-Tc surface at that point
as a proxy for entanglement-entropy curvature — the geometric signal that, in a
real S-AGI, would highlight topological boundaries between superconducting and
non-superconducting phases.

Concretely we compute the (negated) trace of the numerical Hessian of the
surrogate mean. Strongly positive = a peak-like region the loop should
exploit; near-zero = a flat manifold patch; strongly negative = a basin.
"""

from __future__ import annotations

import numpy as np

from .candidates import Candidate, featurize
from .neural import GPSurrogate


def curvature(c: Candidate, model: GPSurrogate, eps: float = 0.05) -> float:
    if model.X_train is None:
        return 0.0
    f0 = featurize(c)
    mu0, _ = model.predict(f0)
    trace = 0.0
    for i in range(len(f0)):
        scale = max(abs(f0[i]), 1.0) * eps
        fp = f0.copy(); fp[i] += scale
        fm = f0.copy(); fm[i] -= scale
        mp, _ = model.predict(fp)
        mm, _ = model.predict(fm)
        trace += (mp[0] + mm[0] - 2.0 * mu0[0]) / (scale ** 2)
    return float(-trace)


def manifold_bonus(c: Candidate, model: GPSurrogate, weight: float = 0.5) -> float:
    """Acquisition bonus from manifold curvature.

    Positive curvature (peak-like) gets rewarded; negative (basin) penalised.
    Weight controls how much this nudges UCB rankings.
    """
    return weight * curvature(c, model)
