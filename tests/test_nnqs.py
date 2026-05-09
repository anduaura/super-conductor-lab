import numpy as np

from scl.nnqs import RBMWavefunction, exact_ground_energy, quantum_proxy
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
