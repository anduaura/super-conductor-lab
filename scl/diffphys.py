"""Differentiable physics: inverse design via gradient descent on the surrogate.

Given a target Tc, we descend in 7-D feature space toward a point the surrogate
predicts will hit it. The result is then *projected* back to a discrete
(composition, pressure) candidate by closest-element-mix matching, and run
through the symbolic verifier before being offered to the loop.

This is the AGI's 'inverse mode': instead of asking 'what does this material
do?', it asks 'what material would do *this*?'.
"""

from __future__ import annotations

import numpy as np

from .candidates import Candidate, ELEMENTS, METALS, featurize, sample_random
from .neural import GPSurrogate
from .symbolic import symbolic_check


_METAL_FEAT = {
    m: np.array([
        ELEMENTS[m]["mass"],
        ELEMENTS[m]["EN"],
        ELEMENTS[m]["radius"],
        ELEMENTS[m]["valence"],
    ], dtype=float)
    for m in METALS
}
_H_FEAT = np.array([
    ELEMENTS["H"]["mass"],
    ELEMENTS["H"]["EN"],
    ELEMENTS["H"]["radius"],
    ELEMENTS["H"]["valence"],
], dtype=float)


def _project(f: np.ndarray) -> Candidate | None:
    avg_mass, avg_en, avg_radius, avg_val, h_frac, _en_diff, pressure = f
    h_frac = float(np.clip(h_frac, 0.06, 0.98))
    pressure = float(np.clip(pressure, 5.0, 595.0))

    target = np.array([avg_mass, avg_en, avg_radius, avg_val], dtype=float)
    metal_target = (target - h_frac * _H_FEAT) / max(1.0 - h_frac, 1e-3)

    best_err = np.inf
    best: tuple[tuple[str, float], ...] | None = None

    for m in METALS:
        err = float(np.linalg.norm(_METAL_FEAT[m] - metal_target))
        if err < best_err:
            best_err = err
            best = ((m, 1.0 - h_frac), ("H", h_frac))

    rem = 1.0 - h_frac
    for i, m1 in enumerate(METALS):
        f1 = _METAL_FEAT[m1]
        for m2 in METALS[i + 1:]:
            f2 = _METAL_FEAT[m2]
            denom = float(np.dot(f1 - f2, f1 - f2)) + 1e-9
            alpha = float(np.clip(
                np.dot(metal_target - f2, f1 - f2) / denom, 0.05, 0.95
            ))
            mix = alpha * f1 + (1.0 - alpha) * f2
            err = float(np.linalg.norm(mix - metal_target))
            if err < best_err:
                best_err = err
                best = (
                    (m1, alpha * rem),
                    (m2, (1.0 - alpha) * rem),
                    ("H", h_frac),
                )

    if best is None:
        return None
    return Candidate(composition=best, pressure_gpa=pressure)


def inverse_design(
    target_tc: float,
    model: GPSurrogate,
    rng: np.random.Generator,
    n_starts: int = 6,
    steps: int = 60,
    lr: float = 0.15,
) -> Candidate | None:
    """Descend toward `target_tc` from several random seeds, keep best valid hit."""
    if model.X_train is None:
        return None

    best_c: Candidate | None = None
    best_err = np.inf

    for _ in range(n_starts):
        seed_c = sample_random(rng)
        f = featurize(seed_c).astype(float)

        for _ in range(steps):
            mu, _ = model.predict(f)
            err = float(mu[0]) - target_tc
            grad = np.zeros_like(f)
            eps_v = np.maximum(np.abs(f) * 1e-3, 1e-3)
            for i in range(len(f)):
                fp = f.copy(); fp[i] += eps_v[i]
                fm = f.copy(); fm[i] -= eps_v[i]
                mp, _ = model.predict(fp)
                mm, _ = model.predict(fm)
                grad[i] = (mp[0] - mm[0]) / (2.0 * eps_v[i])
            denom = float((grad ** 2).sum()) + 1e-6
            f -= lr * err * grad / denom

        c = _project(f)
        if c is None or not symbolic_check(c).ok:
            continue
        mu, _ = model.predict(featurize(c))
        e = abs(float(mu[0]) - target_tc)
        if e < best_err:
            best_err = e
            best_c = c

    return best_c
