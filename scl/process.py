"""Process-engineering layer.

Models the gap between *requested* and *realized* candidates: even when a
synthesis attempt succeeds, the actual nucleated phase may drift in
composition due to quench kinetics, contamination, and pressure-cell limits.
The loop must learn to favor candidates that are *makeable*, not just *good*.
"""

from __future__ import annotations

import numpy as np

from .candidates import Candidate, perturb


def synthesis_window(c: Candidate) -> float:
    """Probability that a synthesis attempt produces a usable sample."""
    p = c.pressure_gpa
    base = 0.95 - 0.6 * max(0.0, (p - 200.0) / 400.0) ** 2
    # Very low pressure also unreliable (poor diffusion).
    if p < 30.0:
        base -= 0.3
    return float(np.clip(base, 0.05, 0.99))


def realized_phase(c: Candidate, rng: np.random.Generator) -> Candidate:
    """Return the actual nucleated phase (may drift from requested)."""
    drift_prob = 0.20 + (0.30 if c.pressure_gpa > 300.0 else 0.0)
    if rng.random() < drift_prob:
        return perturb(c, rng, scale=0.04)
    return c
