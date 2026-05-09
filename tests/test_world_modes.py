import numpy as np

from scl.candidates import Candidate, ELEMENTS, sample_random
from scl.world_model import _PEAKS_MULTI, true_tc


def test_single_mode_unchanged():
    optimum = Candidate(
        composition=(("La", 0.075), ("S", 0.075), ("H", 0.85)),
        pressure_gpa=300.0,
    )
    off = Candidate(composition=(("La", 0.5), ("H", 0.5)), pressure_gpa=50.0)
    assert true_tc(optimum) > true_tc(off) + 50.0


def test_multi_mode_returns_value():
    c = Candidate(composition=(("La", 0.15), ("H", 0.85)), pressure_gpa=200.0)
    val = true_tc(c, mode="multi")
    assert np.isfinite(val) and val >= 0


def test_multi_mode_has_a_higher_peak_than_single_minimum():
    """Random sampling under multi mode should still hit some non-trivial values."""
    rng = np.random.default_rng(0)
    samples = [sample_random(rng) for _ in range(200)]
    multi = [true_tc(c, mode="multi") for c in samples]
    assert max(multi) > 50.0


def test_multi_mode_unknown_raises():
    c = Candidate(composition=(("La", 0.15), ("H", 0.85)), pressure_gpa=200.0)
    try:
        true_tc(c, mode="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown mode")


def test_multi_mode_peak_locations_are_distinct():
    """Sanity: each peak in _PEAKS_MULTI sits at a different (h, p, e, v) point."""
    centers = set()
    for cx_h, cx_p, cx_e, cx_v, *_ in _PEAKS_MULTI:
        key = (round(cx_h, 2), round(cx_p, 0), round(cx_e, 1), round(cx_v, 1))
        centers.add(key)
    assert len(centers) == len(_PEAKS_MULTI)
