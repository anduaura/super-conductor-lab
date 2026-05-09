import numpy as np

from scl.candidates import Candidate, sample_random, featurize
from scl.manifold import curvature, manifold_bonus
from scl.neural import GPSurrogate


def test_curvature_zero_without_data():
    c = Candidate(composition=(("La", 0.15), ("H", 0.85)), pressure_gpa=200.0)
    gp = GPSurrogate()
    assert curvature(c, gp) == 0.0


def test_curvature_finite_after_training():
    rng = np.random.default_rng(0)
    samples = [sample_random(rng) for _ in range(15)]
    X = np.stack([featurize(c) for c in samples])
    y = X[:, 4] * 100.0  # arbitrary smooth target
    gp = GPSurrogate()
    gp.fit(X, y)
    c = samples[0]
    val = curvature(c, gp)
    assert np.isfinite(val)


def test_manifold_bonus_scales_with_weight():
    rng = np.random.default_rng(0)
    samples = [sample_random(rng) for _ in range(15)]
    X = np.stack([featurize(c) for c in samples])
    y = X[:, 4] * 100.0
    gp = GPSurrogate()
    gp.fit(X, y)
    a = manifold_bonus(samples[0], gp, weight=0.5)
    b = manifold_bonus(samples[0], gp, weight=1.0)
    assert np.isclose(b, 2.0 * a, atol=1e-9)
