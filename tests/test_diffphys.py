import numpy as np

from scl.candidates import Candidate, featurize, sample_random
from scl.diffphys import inverse_design
from scl.neural import GPSurrogate
from scl.symbolic import symbolic_check


def test_inverse_design_returns_none_with_unfit_model():
    rng = np.random.default_rng(0)
    gp = GPSurrogate()
    out = inverse_design(target_tc=300.0, model=gp, rng=rng)
    assert out is None


def test_inverse_design_returns_valid_candidate():
    rng = np.random.default_rng(0)
    samples = [sample_random(rng) for _ in range(15)]
    X = np.stack([featurize(c) for c in samples])
    # Construct a target where the surrogate has clear structure.
    y = 100.0 + 200.0 * X[:, 4]  # higher H fraction → higher Tc
    gp = GPSurrogate()
    gp.fit(X, y)

    rng2 = np.random.default_rng(1)
    out = inverse_design(target_tc=250.0, model=gp, rng=rng2,
                         n_starts=4, steps=30)
    assert out is not None
    assert isinstance(out, Candidate)
    assert symbolic_check(out).ok
