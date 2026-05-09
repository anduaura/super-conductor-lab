"""Gaussian-process surrogate ('System 1' hunch).

Stands in for the GNN/transformer family in a real S-AGI: a function from a
composition's feature vector to a posterior over Tc. Uncertainty is what makes
active learning possible — the loop intentionally probes regions where this
model claims to be unsure.
"""

from __future__ import annotations

import numpy as np


class GPSurrogate:
    def __init__(
        self,
        lengthscale: float = 1.0,
        signal_var: float = 2500.0,
        noise_var: float = 25.0,
    ):
        self.lengthscale = lengthscale
        self.signal_var = signal_var
        self.noise_var = noise_var
        self.X_train: np.ndarray | None = None
        self.L: np.ndarray | None = None
        self.alpha: np.ndarray | None = None
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: float = 0.0

    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        sq = (
            np.sum(A ** 2, axis=1)[:, None]
            + np.sum(B ** 2, axis=1)[None, :]
            - 2.0 * A @ B.T
        )
        sq = np.maximum(sq, 0.0)
        return self.signal_var * np.exp(-0.5 * sq / (self.lengthscale ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y row counts disagree")

        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0)
        self.x_std = np.where(self.x_std < 1e-9, 1.0, self.x_std)
        Xs = (X - self.x_mean) / self.x_std

        self.y_mean = float(y.mean())
        ys = y - self.y_mean

        K = self._kernel(Xs, Xs) + self.noise_var * np.eye(len(Xs))
        # Jitter for numerical stability if noise_var is small.
        self.L = np.linalg.cholesky(K + 1e-8 * np.eye(len(Xs)))
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, ys))
        self.X_train = Xs

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]

        if self.X_train is None:
            n = X.shape[0]
            return (
                np.full(n, self.y_mean),
                np.full(n, float(np.sqrt(self.signal_var + self.noise_var))),
            )

        Xs = (X - self.x_mean) / self.x_std
        Ks = self._kernel(self.X_train, Xs)  # (N_train, N_test)
        mean = Ks.T @ self.alpha + self.y_mean

        v = np.linalg.solve(self.L, Ks)
        var = self.signal_var - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 0.0) + self.noise_var
        return mean, np.sqrt(var)
