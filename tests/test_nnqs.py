import math

import numpy as np

from scl.nnqs import (
    RBMWavefunction,
    exact_ground_energy,
    hubbard_ground_energy,
    hubbard_proxy,
    quantum_proxy,
)
from scl.candidates import Candidate


def test_exact_diag_known_limits():
    # h=0 limit: classical ferromagnet, E_gs = -J * N (PBC).
    e0 = exact_ground_energy(n_sites=4, J=1.0, h=0.0)
    assert np.isclose(e0, -4.0, atol=1e-9)
    # J=0 limit: free transverse field, E_gs = -h * N.
    e1 = exact_ground_energy(n_sites=4, J=0.0, h=1.0)
    assert np.isclose(e1, -4.0, atol=1e-9)


def test_rbm_recovers_classical_limit():
    # h=0: RBM should converge close to -J*N within a generous tolerance.
    rbm = RBMWavefunction(n_sites=4, n_hidden=4, seed=0)
    rbm.fit(J=1.0, h=0.0, steps=200, lr=0.05)
    e = rbm.energy(J=1.0, h=0.0)
    exact = exact_ground_energy(4, 1.0, 0.0)
    assert e <= exact + 0.5  # variational upper bound, within 0.5 of exact


def test_rbm_close_to_exact_at_critical_point():
    # h ≈ J critical point: RBM should still produce a sensible upper bound.
    rbm = RBMWavefunction(n_sites=4, n_hidden=6, seed=1)
    rbm.fit(J=1.0, h=1.0, steps=400, lr=0.05)
    e = rbm.energy(J=1.0, h=1.0)
    exact = exact_ground_energy(4, 1.0, 1.0)
    # Variational principle: e >= exact. Allow up to 25% gap (RBM is small).
    assert e >= exact - 1e-6
    assert e <= exact * 0.75 if exact < 0 else True


def test_quantum_proxy_is_bounded():
    c = Candidate(composition=(("La", 0.15), ("H", 0.85)), pressure_gpa=200.0)
    e = quantum_proxy(c, n_sites=4, n_hidden=4, steps=40, lr=0.05)
    # Per-site energy should be O(1) for our (J, h) regime.
    assert -10.0 < e < 10.0


# ---- Hubbard exact-diag calibration ---------------------------------------


def test_hubbard_t_zero_atomic_limit_is_zero():
    """At t=0, half-filling, ground state has zero double occupancy → E=0."""
    assert hubbard_ground_energy(4, t=0.0, U=1.0, periodic=True) == 0.0
    assert hubbard_ground_energy(4, t=0.0, U=5.0, periodic=False) == 0.0
    assert hubbard_ground_energy(6, t=0.0, U=2.0, periodic=True) == 0.0


def test_hubbard_u_zero_matches_free_fermion_obc():
    """At U=0, ground energy must equal the sum of the lowest single-particle
    energies. For N=4 OBC: ε_n = -2t cos(nπ/(N+1)); fill the lowest 2 per
    spin sector."""
    expected = 2 * (-2 * math.cos(math.pi / 5) - 2 * math.cos(2 * math.pi / 5))
    got = hubbard_ground_energy(4, t=1.0, U=0.0, periodic=False)
    assert abs(got - expected) < 1e-9


def test_hubbard_kinetic_energy_grows_monotonically_with_t():
    """For fixed U, doubling t at least doubles |E_gs| (kinetic dominates)."""
    e_low = hubbard_ground_energy(4, t=0.5, U=1.0)
    e_high = hubbard_ground_energy(4, t=2.0, U=1.0)
    assert abs(e_high) > 2 * abs(e_low)


def test_hubbard_energy_grows_monotonically_with_u():
    """For fixed t, increasing U raises the ground energy (electrons less
    free to delocalise)."""
    e_low = hubbard_ground_energy(4, t=1.0, U=0.0)
    e_high = hubbard_ground_energy(4, t=1.0, U=4.0)
    assert e_high > e_low


def test_hubbard_proxy_returns_finite_value():
    c = Candidate(composition=(("La", 0.15), ("H", 0.85)), pressure_gpa=200.0)
    e = hubbard_proxy(c, n_sites=4)
    assert np.isfinite(e)
    # Per-site energy in our (t, U) regime should be O(1).
    assert -5.0 < e < 5.0


def test_hubbard_proxy_responds_to_h_fraction():
    """Higher H fraction → larger t → more negative kinetic energy."""
    low_h = Candidate(composition=(("La", 0.50), ("H", 0.50)), pressure_gpa=200.0)
    high_h = Candidate(composition=(("La", 0.15), ("H", 0.85)), pressure_gpa=200.0)
    e_low = hubbard_proxy(low_h)
    e_high = hubbard_proxy(high_h)
    # More H → larger t → lower (more negative) per-site energy.
    assert e_high < e_low


def test_hubbard_proxy_responds_to_en_contrast():
    """Higher EN spread → larger U → less negative ground energy
    (Coulomb pushes back against delocalisation)."""
    low_en = Candidate(
        composition=(("La", 0.15), ("Y", 0.10), ("H", 0.75)),
        pressure_gpa=200.0,
    )  # EN spread small (1.10, 1.22, 2.20)
    high_en = Candidate(
        composition=(("Li", 0.15), ("S", 0.10), ("H", 0.75)),
        pressure_gpa=200.0,
    )  # EN spread large (0.98, 2.58, 2.20)
    e_low = hubbard_proxy(low_en)
    e_high_en = hubbard_proxy(high_en)
    assert e_high_en > e_low
