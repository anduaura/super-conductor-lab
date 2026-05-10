"""Chemistry helpers grounded in the ``scl.candidates.ELEMENTS`` table.

These are the building blocks that ``scl.symbolic`` uses for stricter
chemistry checks than the original toy rules. Functions here are
*deterministic and pure* — they compute scores from a candidate's
composition, never call out to remote APIs.

When the optional ``[chem]`` extra is installed (``pip install -e
'.[chem]'``), additional pymatgen-backed checks become available via
``pymatgen_validator`` — see ``scl.symbolic`` for the registered rule.
"""

from __future__ import annotations

import math
from typing import Optional

from .candidates import Candidate, ELEMENTS


# ----------------------------------------------------------------------------
# Pure-numpy chemistry rules
# ----------------------------------------------------------------------------


def charge_residual(c: Candidate) -> float:
    """Charge residual = sum of (formal valence × fraction).

    A charge-balanced composition has residual ≈ 0. Hydrides typically have
    a small positive residual (H is +1 nominally but can act as H⁻ with a
    transition metal); we tolerate up to ~|0.5|.
    """
    return sum(f * ELEMENTS[e]["valence"] for e, f in c.composition)


def hydrogen_metal_ratio(c: Candidate) -> Optional[float]:
    """H atom count divided by total non-H atom count. Returns None for
    pure-H or no-H compositions."""
    h = next((f for e, f in c.composition if e == "H"), 0.0)
    metals = sum(f for e, f in c.composition if e != "H")
    if metals <= 1e-9 or h <= 1e-9:
        return None
    return float(h / metals)


def formation_driving_force(c: Candidate) -> float:
    """Pauling-style formation-enthalpy proxy.

    For each pair (i, j) of distinct elements in the composition, the
    contribution to ΔH_f is approximately ``-(EN_i - EN_j)² × x_i × x_j``
    (negative = stabilising). We return the magnitude — larger means a
    stronger thermodynamic driving force to form the compound.

    This is a *very* coarse proxy; real ΔH_f requires DFT. It's calibrated
    only well enough to discriminate "no driving force at all" from "yes,
    these atoms want to bond."
    """
    items = list(c.composition)
    total = 0.0
    for i, (ei, xi) in enumerate(items):
        en_i = ELEMENTS[ei]["EN"]
        for ej, xj in items[i + 1:]:
            en_j = ELEMENTS[ej]["EN"]
            total += xi * xj * (en_i - en_j) ** 2
    return float(total)


def electron_count_per_formula(c: Candidate) -> float:
    """Mean number of valence electrons per formula unit (weighted by Z).

    Sanity proxy — extreme values suggest unusual ionization states. Most
    real superconductors sit between 4–8 valence electrons per cation site.
    """
    total = 0.0
    for e, f in c.composition:
        z = ELEMENTS[e]["Z"]
        # Crude valence electron count from atomic number (groups 1-2 + 13-18).
        if z == 1:
            ve = 1
        elif z <= 4:
            ve = z
        elif z <= 10:
            ve = z - 2
        elif z <= 12:
            ve = z - 10
        elif z <= 18:
            ve = z - 10
        elif z <= 20:
            ve = z - 18
        elif z <= 38:
            ve = (z - 18) if z <= 20 else max(2, z - 20 + 2)
        else:
            ve = 3  # Lanthanides nominally trivalent
        total += f * ve
    return float(total)


# ----------------------------------------------------------------------------
# Optional pymatgen integration (gated by [chem] extra)
# ----------------------------------------------------------------------------


def pymatgen_available() -> bool:
    try:
        import pymatgen  # noqa: F401
    except ImportError:
        return False
    return True


def pymatgen_charge_balanced(c: Candidate, tol: float = 0.05) -> tuple[bool, str]:
    """Validate charge balance using pymatgen's Composition class.

    Returns (ok, message). Falls back to the in-house ``charge_residual``
    if pymatgen isn't installed — the message indicates the source.
    """
    try:
        from pymatgen.core import Composition  # type: ignore
    except ImportError:
        residual = charge_residual(c)
        ok = abs(residual) < tol
        return ok, f"in-house charge residual {residual:+.3f} (pymatgen not installed)"

    # Build a pymatgen Composition by scaling fractions to a common base.
    # Multiply by a large factor to get integer-ish counts.
    counts = {e: f * 100.0 for e, f in c.composition}
    comp = Composition(counts)
    # pymatgen's oxi_state_guesses: list of dicts of element→oxidation state
    # that balance to zero. If non-empty, the composition is balanceable.
    guesses = comp.oxi_state_guesses(target_charge=0)
    if guesses:
        return True, f"pymatgen finds {len(guesses)} valid oxidation-state assignment(s)"
    return False, "pymatgen finds no charge-balanced oxidation states"
