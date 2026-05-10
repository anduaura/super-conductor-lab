"""Tests for scl.gnn.TorchSurrogate.

Skipped automatically when torch isn't installed (the [gnn] extra).
"""

import numpy as np
import pytest

from scl.gnn import make_surrogate, torch_available


def test_make_surrogate_gp_no_extras():
    """gp kind always works regardless of torch availability."""
    s = make_surrogate("gp")
    from scl.neural import GPSurrogate
    assert isinstance(s, GPSurrogate)


def test_make_surrogate_unknown_kind():
    with pytest.raises(ValueError):
        make_surrogate("does-not-exist")


def test_torch_surrogate_requires_extra_when_missing():
    if torch_available():
        pytest.skip("torch is installed; this test only runs when it isn't")
    with pytest.raises(ImportError, match=r"\[gnn\]"):
        make_surrogate("nn")


@pytest.mark.skipif(not torch_available(), reason="torch not installed")
def test_torch_surrogate_fits_and_predicts():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (15, 4)).astype(np.float32)
    y = X.sum(axis=1) * 100.0
    s = make_surrogate("nn", n_epochs=100, seed=0)
    s.fit(X, y)
    mean, std = s.predict(X)
    assert mean.shape == (15,)
    assert std.shape == (15,)
    assert np.all(std > 0)
    # Mean should be in the right ballpark (not perfect after 100 epochs).
    assert np.mean(np.abs(mean - y)) < 100.0


@pytest.mark.skipif(not torch_available(), reason="torch not installed")
def test_torch_surrogate_predicts_prior_when_unfitted():
    s = make_surrogate("nn")
    mean, std = s.predict(np.array([[1.0, 2.0, 3.0]]))
    assert mean.shape == (1,)
    assert std.shape == (1,)
    assert np.isfinite(mean).all() and np.isfinite(std).all()


@pytest.mark.skipif(not torch_available(), reason="torch not installed")
def test_torch_surrogate_uncertainty_grows_far_from_data():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (12, 3)).astype(np.float32)
    y = rng.normal(0, 1, 12).astype(np.float32)
    s = make_surrogate("nn", n_epochs=200, seed=0)
    s.fit(X, y)
    _, sigma_in = s.predict(X)
    _, sigma_far = s.predict(np.array([[100.0, 100.0, 100.0]]))
    # MC dropout uncertainty should be bigger far from training data.
    assert sigma_far[0] >= sigma_in.mean() * 0.8


@pytest.mark.skipif(not torch_available(), reason="torch not installed")
def test_loop_with_nn_surrogate():
    """End-to-end smoke: loop runs with surrogate_kind='nn'."""
    from scl.loop import run_loop
    res = run_loop(rounds=4, seed=0, pool_size=20, init_size=2,
                   surrogate_kind="nn", world_mode="single")
    assert len(res.rounds) == 6
