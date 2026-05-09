import numpy as np

from scl.candidates import Candidate
from scl.lab import Lab
from scl.process import realized_phase, synthesis_window


def test_synthesis_window_decreases_with_pressure():
    low = Candidate(composition=(("La", 0.15), ("H", 0.85)), pressure_gpa=100.0)
    high = Candidate(composition=(("La", 0.15), ("H", 0.85)), pressure_gpa=550.0)
    assert synthesis_window(low) > synthesis_window(high)


def test_synthesis_window_in_unit_interval():
    for p in (5.0, 100.0, 200.0, 400.0, 600.0):
        c = Candidate(composition=(("La", 0.15), ("H", 0.85)), pressure_gpa=p)
        w = synthesis_window(c)
        assert 0.0 <= w <= 1.0


def test_realized_phase_sometimes_drifts():
    """High-pressure candidates should drift more often than low-pressure."""
    rng_lo = np.random.default_rng(0)
    rng_hi = np.random.default_rng(0)
    lo = Candidate(composition=(("La", 0.15), ("H", 0.85)), pressure_gpa=100.0)
    hi = Candidate(composition=(("La", 0.15), ("H", 0.85)), pressure_gpa=400.0)
    drift_lo = sum(1 for _ in range(200) if realized_phase(lo, rng_lo) is not lo)
    drift_hi = sum(1 for _ in range(200) if realized_phase(hi, rng_hi) is not hi)
    assert drift_hi > drift_lo


def test_lab_returns_realized_candidate_field():
    rng = np.random.default_rng(0)
    lab = Lab(rng=rng)
    c = Candidate(composition=(("La", 0.15), ("H", 0.85)), pressure_gpa=200.0)
    seen_drift = False
    for _ in range(50):
        m = lab.run(c)
        assert m.requested is c
        if m.success and m.candidate is not c:
            seen_drift = True
    # Over 50 attempts at moderate pressure we should observe at least one drift.
    assert seen_drift
