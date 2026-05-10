"""Tests for scl.chem and the chem-aware symbolic rules."""

import pytest

from scl import chem
from scl.candidates import Candidate
from scl.symbolic import symbolic_check


def test_charge_residual_balanced():
    c = Candidate(composition=(("La", 0.10), ("S", 0.05), ("H", 0.85)),
                  pressure_gpa=200.0)
    # La +3 × 0.10 + S -2 × 0.05 + H +1 × 0.85 = 0.30 - 0.10 + 0.85 = 1.05
    assert abs(chem.charge_residual(c) - 1.05) < 1e-6


def test_charge_residual_zero_for_balanced_oxide_like():
    # Mg+2 ⋅ 0.5 + S-2 ⋅ 0.5 = 0
    c = Candidate(composition=(("Mg", 0.50), ("S", 0.50)), pressure_gpa=10.0)
    assert abs(chem.charge_residual(c)) < 1e-6


def test_hydrogen_metal_ratio_basic():
    # 0.85 H to 0.15 metal → ratio 5.67 (LaH10-ish range when scaled)
    c = Candidate(composition=(("La", 0.15), ("H", 0.85)), pressure_gpa=200.0)
    assert chem.hydrogen_metal_ratio(c) == pytest.approx(0.85 / 0.15, rel=1e-6)


def test_hydrogen_metal_ratio_pure_metal_returns_none():
    c = Candidate(composition=(("La", 1.0),), pressure_gpa=200.0)
    assert chem.hydrogen_metal_ratio(c) is None


def test_formation_driving_force_strong_for_polar_pair():
    # H (EN 2.20) + La (1.10): EN diff = 1.10, x_i x_j = 0.85 * 0.15 = 0.1275
    # contribution = 0.1275 * 1.21 ≈ 0.154
    c = Candidate(composition=(("La", 0.15), ("H", 0.85)), pressure_gpa=200.0)
    f = chem.formation_driving_force(c)
    assert f > 0.1


def test_formation_driving_force_zero_for_pure():
    c = Candidate(composition=(("H", 1.0),), pressure_gpa=10.0)
    assert chem.formation_driving_force(c) == 0.0


def test_symbolic_rejects_unbalanced_charge_via_new_rule():
    """A wildly unbalanced composition (was passing under old |.|<4 rule)
    should now fail the tightened |.|<1.5 charge-balance check."""
    # 4 × +3 metal + 0% counter-anion = +12 (way over)
    c = Candidate(composition=(("Y", 0.50), ("La", 0.50)), pressure_gpa=200.0)
    res = symbolic_check(c)
    failures = {n: msg for n, _, msg in res.failures}
    assert "hydrogen-present" in failures  # also fails — no H
    # Test with H so we isolate the charge rule
    c2 = Candidate(composition=(("Y", 0.45), ("La", 0.45), ("H", 0.10)),
                   pressure_gpa=200.0)
    res2 = symbolic_check(c2)
    names = {n for n, _, _ in res2.failures}
    # avg valence = 0.45*3 + 0.45*3 + 0.1*1 = 2.8 → fails |.|<1.5
    assert "charge-balance" in names


def test_symbolic_rejects_invalid_hydride_ratio():
    """An MH50-like ratio (way too H-rich for any known hydride) should fail
    the new hydride-stoichiometry soft rule."""
    # Make the hydrogen-present rule pass (h_frac < 0.99) while pushing the
    # H:metal ratio above 15 — needs a tiny non-H slice.
    c = Candidate(composition=(("La", 0.05), ("H", 0.95)), pressure_gpa=200.0)
    # 0.95 / 0.05 = 19 → outside [0.5, 15]
    res = symbolic_check(c)
    names = {n for n, _, _ in res.failures}
    assert "hydride-stoichiometry" in names


def test_pymatgen_rule_registered_only_when_available():
    """The pymatgen rule should appear in the registry iff pymatgen imports."""
    from scl.symbolic import _RULES
    names = {n for n, _, _ in _RULES}
    if chem.pymatgen_available():
        assert "pymatgen-charge-balanced" in names
    else:
        assert "pymatgen-charge-balanced" not in names


@pytest.mark.skipif(not chem.pymatgen_available(),
                    reason="pymatgen not installed (install with .[chem])")
def test_pymatgen_charge_balanced_returns_tuple():
    c = Candidate(composition=(("La", 0.15), ("H", 0.85)), pressure_gpa=200.0)
    ok, msg = chem.pymatgen_charge_balanced(c)
    assert isinstance(ok, bool)
    assert "pymatgen" in msg
