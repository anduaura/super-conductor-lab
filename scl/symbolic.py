"""Symbolic rule engine ("System 2").

Rules are deliberately conservative first-principles checks. A 'hard' failure
vetoes the candidate entirely; 'soft' failures are logged but pass through so
the surrogate can still learn from the data point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Tuple

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
    avg_val = sum(f * ELEMENTS[e]["valence"] for e, f in c.composition)
    return abs(avg_val) < 4.0, f"avg valence {avg_val:.2f} indicates large charge imbalance"


@_rule("formation-driving-force", severity="soft")
def _formation(c: Candidate) -> Tuple[bool, str]:
    """No EN spread → no thermodynamic driving force for compound formation."""
    ens = [ELEMENTS[e]["EN"] for e, _ in c.composition]
    spread = max(ens) - min(ens) if len(ens) > 1 else 0.0
    return spread > 0.3, f"electronegativity spread {spread:.2f} below formation threshold"


@_rule("pauli-orbital-density", severity="soft")
def _pauli(c: Candidate) -> Tuple[bool, str]:
    """Pauli-overlap proxy: mean ionic radius must exceed bare H radius."""
    avg_r = sum(f * ELEMENTS[e]["radius"] for e, f in c.composition)
    return avg_r > 30.0, f"mean ionic radius {avg_r:.0f}pm violates Pauli-overlap proxy"


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
