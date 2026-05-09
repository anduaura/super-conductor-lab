import numpy as np
import pytest

from scl.candidates import Candidate, sample_random
from scl.symbolic import symbolic_check


def test_pure_metal_is_vetoed():
    c = Candidate(composition=(("La", 1.0),), pressure_gpa=200.0)
    res = symbolic_check(c)
    assert not res.ok
    assert any(name == "hydrogen-present" for name, _, _ in res.failures)


def test_pure_hydrogen_is_vetoed():
    c = Candidate(composition=(("H", 1.0),), pressure_gpa=200.0)
    assert not symbolic_check(c).ok


def test_negative_pressure_is_vetoed():
    c = Candidate(composition=(("La", 0.15), ("H", 0.85)), pressure_gpa=-10.0)
    assert not symbolic_check(c).ok


def test_typical_lah_passes():
    c = Candidate(composition=(("La", 0.15), ("H", 0.85)), pressure_gpa=200.0)
    assert symbolic_check(c).ok


def test_unknown_element_vetoed():
    c = Candidate(composition=(("Xx", 0.15), ("H", 0.85)), pressure_gpa=200.0)
    assert not symbolic_check(c).ok


def test_random_samples_largely_pass():
    rng = np.random.default_rng(0)
    samples = [sample_random(rng) for _ in range(200)]
    pass_rate = sum(1 for s in samples if symbolic_check(s).ok) / len(samples)
    # By construction sample_random keeps H in (0.10, 0.95) and pressure in (20, 500),
    # so the soft charge-balance rule is the only thing that can sometimes fire,
    # and that's a soft rule — pass rate should be ~1.0.
    assert pass_rate > 0.95
