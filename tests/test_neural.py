import numpy as np

from scl.neural import GPSurrogate


def test_predicts_prior_when_unfitted():
    gp = GPSurrogate()
    mu, sigma = gp.predict(np.array([[1.0, 2.0, 3.0]]))
    assert mu.shape == (1,)
    assert sigma.shape == (1,)
    assert np.isfinite(mu).all() and np.isfinite(sigma).all()


def test_recovers_training_points():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (12, 4))
    y = (X.sum(axis=1) * 50.0)
    gp = GPSurrogate(lengthscale=0.5, signal_var=10000.0, noise_var=1.0)
    gp.fit(X, y)
    mu, _ = gp.predict(X)
    # With small noise the posterior mean at training points is close to y.
    assert np.mean(np.abs(mu - y)) < 5.0


def test_uncertainty_grows_far_from_data():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (8, 3))
    y = rng.normal(0, 1, 8)
    gp = GPSurrogate(lengthscale=0.5)
    gp.fit(X, y)
    _, sigma_in = gp.predict(X)
    # A point ~10 standardized units from training data should be far less certain.
    far = np.array([[100.0, 100.0, 100.0]])
    _, sigma_far = gp.predict(far)
    assert sigma_far[0] > sigma_in.mean()
