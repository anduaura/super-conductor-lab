"""Hidden ground-truth Tc landscape (the 'real world' for this simulator).

Two modes:

* ``single`` — the original peaked landscape: optimal H ≈ 0.85, EN contrast
  ≈ 1.5, valence ≈ 1.5, pressure helps with diminishing returns. Smooth
  unimodal — UCB will dominate it.
* ``multi`` — sum of four Gaussian peaks at different (h_frac, pressure,
  en_diff, avg_val) combinations with different heights and basin widths.
  The highest peak (~320 K) sits in a narrow attractor at an unusual
  valence; the easiest peak (~220 K) covers far more of the search volume.
  Designed to expose the difference between the acquisition strategies.

The surrogate (``scl.neural``) never imports this module directly — only
``scl.lab`` (and tests) call ``true_tc``.
"""

from __future__ import annotations

import numpy as np

from .candidates import Candidate, featurize


_BASE_TC_K = 350.0


def _single_tc(c: Candidate) -> float:
    feats = featurize(c)
    avg_mass, avg_en, avg_radius, avg_val, h_frac, en_diff, pressure = feats

    h_term = float(np.exp(-((h_frac - 0.85) ** 2) / 0.02))
    en_term = float(np.exp(-((en_diff - 1.5) ** 2) / 0.5))
    sym_term = float(np.exp(-((avg_val - 1.5) ** 2) / 2.0))
    p_term = float(pressure / (pressure + 50.0))
    mass_penalty = float(np.exp(-avg_mass / 500.0))

    return _BASE_TC_K * h_term * en_term * sym_term * p_term * mass_penalty


# Each peak: (h_frac, pressure_gpa, en_diff, avg_val,
#             σ_h, σ_p, σ_e, σ_v, height_K).
_PEAKS_MULTI = (
    (0.80, 100.0, 0.8, 1.5, 0.06, 60.0, 0.4, 0.6, 220.0),
    (0.85, 200.0, 1.2, 1.8, 0.04, 60.0, 0.3, 0.5, 270.0),
    (0.90, 400.0, 0.6, 2.0, 0.04, 80.0, 0.4, 0.6, 260.0),
    (0.85, 280.0, 1.4, 0.5, 0.03, 50.0, 0.3, 0.4, 320.0),
)


def _multi_tc(c: Candidate) -> float:
    feats = featurize(c)
    _, _, _, avg_val, h_frac, en_diff, pressure = feats
    total = 0.0
    for cx_h, cx_p, cx_e, cx_v, sx_h, sx_p, sx_e, sx_v, height in _PEAKS_MULTI:
        d2 = (
            ((h_frac - cx_h) / sx_h) ** 2
            + ((pressure - cx_p) / sx_p) ** 2
            + ((en_diff - cx_e) / sx_e) ** 2
            + ((avg_val - cx_v) / sx_v) ** 2
        )
        total += height * float(np.exp(-0.5 * d2))
    return total


def true_tc(c: Candidate, mode: str = "single") -> float:
    if mode == "single":
        return _single_tc(c)
    if mode == "multi":
        return _multi_tc(c)
    raise ValueError(f"unknown world model mode: {mode!r}")


WORLD_MODES = ("single", "multi")
