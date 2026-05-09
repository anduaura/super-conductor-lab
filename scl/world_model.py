"""Hidden ground-truth Tc landscape (the 'real world' for this simulator).

Stands in for what a full DFT + Eliashberg pipeline would estimate. The
surrogate (scl.neural) never sees this directly — only the noisy lab readings
of it. The function is constructed so that:

  - Optimal H fraction sits near 0.85 (the superhydride regime).
  - A moderate electronegativity contrast (~1.5) is rewarded — the
    ionic-covalent mix that helps phonon-mediated coupling.
  - Average valence near 1.5 is rewarded as a crude symmetry proxy.
  - Pressure helps with diminishing returns.
"""

from __future__ import annotations

import numpy as np

from .candidates import Candidate, featurize


_BASE_TC_K = 350.0  # ceiling of the landscape, in Kelvin


def true_tc(c: Candidate) -> float:
    feats = featurize(c)
    avg_mass, avg_en, avg_radius, avg_val, h_frac, en_diff, pressure = feats

    h_term = float(np.exp(-((h_frac - 0.85) ** 2) / 0.02))
    en_term = float(np.exp(-((en_diff - 1.5) ** 2) / 0.5))
    sym_term = float(np.exp(-((avg_val - 1.5) ** 2) / 2.0))
    p_term = float(pressure / (pressure + 50.0))

    # heavy lattices weakly suppress phonon frequencies — small mass penalty.
    mass_penalty = float(np.exp(-avg_mass / 500.0))

    return _BASE_TC_K * h_term * en_term * sym_term * p_term * mass_penalty
