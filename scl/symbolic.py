"""Symbolic rule engine ("System 2").

Rules are deliberately conservative first-principles checks. A 'hard' failure
vetoes the candidate entirely; 'soft' failures are logged but pass through so
the surrogate can still learn from the data point.

Chemistry rules (charge balance, formation driving force, hydride
stoichiometry, electron count) defer to the helpers in ``scl.chem``. When
the optional ``[chem]`` extra (pymatgen) is installed, an additional
pymatgen-validated rule runs alongside the in-house checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Tuple

from . import chem
from .candidates import Candidate, ELEMENTS


Severity = str  # "hard" or "soft"
RuleFn = Callable[[Candidate], Tuple[bool, str]]
_RULES: List[Tuple[str, Severity, RuleFn]] = []


def _rule(name: str, severity: Severity = "hard"):
    def decorate(fn: RuleFn) -> RuleFn:
        _RULES.append((name, severity, fn))
        return fn
    return decorate


@_rule("fractions-sum-to-one")
def _frac_sum(c: Candidate) -> Tuple[bool, str]:
    s = sum(f for _, f in c.composition)
    return abs(s - 1.0) < 1e-6, f"composition fractions sum to {s:.6f}, expected 1.0"


@_rule("fractions-positive")
def _frac_pos(c: Candidate) -> Tuple[bool, str]:
    bad = [(e, f) for e, f in c.composition if f <= 0.0 or f >= 1.0001]
    return not bad, f"non-physical fractions: {bad}"


@_rule("hydrogen-present")
def _h_present(c: Candidate) -> Tuple[bool, str]:
    h = c.h_fraction()
    return 0.05 < h < 0.99, f"H fraction {h:.3f} outside (0.05, 0.99)"


@_rule("pressure-bounds")
def _pressure(c: Candidate) -> Tuple[bool, str]:
    p = c.pressure_gpa
    return 0.0 < p <= 600.0, f"pressure {p:.1f} GPa outside (0, 600]"


@_rule("known-elements")
def _known(c: Candidate) -> Tuple[bool, str]:
    unknown = [e for e, _ in c.composition if e not in ELEMENTS]
    return not unknown, f"unknown elements: {unknown}"


@_rule("charge-balance", severity="soft")
def _charge(c: Candidate) -> Tuple[bool, str]:
    """Sum of formal valences × fractions should be near zero.

    Tightened from the original |Σ| < 4 to |Σ| < 1.5 — hydrides have small
    residuals from H acting as both H⁺ and H⁻, but anything beyond ~1.5
    indicates the composition genuinely doesn't balance.
    """
    residual = chem.charge_residual(c)
    return abs(residual) < 1.5, f"charge residual {residual:+.2f} (need |.| < 1.5)"


@_rule("formation-driving-force", severity="soft")
def _formation(c: Candidate) -> Tuple[bool, str]:
    """Pauling-style formation-enthalpy proxy must exceed a minimum.

    Stricter than the original "EN spread > 0.3" check: requires actual
    pair-wise (EN_i - EN_j)² × x_i × x_j contributions to be non-trivial.
    """
    f_proxy = chem.formation_driving_force(c)
    return f_proxy > 0.05, f"formation driving force {f_proxy:.3f} below threshold"


@_rule("hydride-stoichiometry", severity="soft")
def _hydride_ratio(c: Candidate) -> Tuple[bool, str]:
    """H : metal ratio should match known hydride stoichiometries.

    Practical superhydrides span MH (~1.0) up to MH₁₂ (~12.0). Beyond ~15,
    the H sublattice can't be coordinated by the metal sublattice; below
    0.5 it's not really a hydride.
    """
    r = chem.hydrogen_metal_ratio(c)
    if r is None:
        return True, "no H-metal ratio applicable"
    return 0.5 <= r <= 15.0, f"H:metal ratio {r:.2f} outside typical hydride range [0.5, 15]"


@_rule("pauli-orbital-density", severity="soft")
def _pauli(c: Candidate) -> Tuple[bool, str]:
    """Pauli-overlap proxy: mean ionic radius must exceed bare H radius."""
    avg_r = sum(f * ELEMENTS[e]["radius"] for e, f in c.composition)
    return avg_r > 30.0, f"mean ionic radius {avg_r:.0f}pm violates Pauli-overlap proxy"


# Optional pymatgen-backed rule. Only registered when pymatgen is importable
# (i.e. user installed with `pip install -e '.[chem]'`).
if chem.pymatgen_available():
    @_rule("pymatgen-charge-balanced", severity="soft")
    def _pymatgen_charge(c: Candidate) -> Tuple[bool, str]:
        return chem.pymatgen_charge_balanced(c)


@dataclass
class SymbolicResult:
    ok: bool
    failures: List[Tuple[str, Severity, str]] = field(default_factory=list)

    def hard_failures(self) -> List[Tuple[str, str]]:
        return [(name, msg) for name, sev, msg in self.failures if sev == "hard"]


def symbolic_check(c: Candidate) -> SymbolicResult:
    """Run all rules, return aggregate result. ok=False iff any *hard* rule fires."""
    fails: List[Tuple[str, Severity, str]] = []
    for name, severity, fn in _RULES:
        try:
            ok, msg = fn(c)
        except Exception as exc:
            ok, msg = False, f"rule raised: {exc!r}"
        if not ok:
            fails.append((name, severity, msg))
    hard_fired = any(sev == "hard" for _, sev, _ in fails)
    return SymbolicResult(ok=not hard_fired, failures=fails)
