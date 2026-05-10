"""Torch-backed neural surrogate — drop-in replacement for ``scl.neural.GPSurrogate``.

Behind the ``[gnn]`` optional extra (``pip install -e '.[gnn]'``). Provides
the same ``fit(X, y)`` / ``predict(X) -> (mean, std)`` contract so it can be
swapped in via the ``surrogate_kind`` parameter on ``scl.loop.run_loop``
without any other code changes.

**Today's implementation** is a small MLP with MC dropout for epistemic
uncertainty. Operates on the existing 7-feature composition vector.

**Tomorrow's implementation** is a graph neural network over crystal
structures, transfer-learned from a public dataset (Materials Project's
superconductor subset, ~16K labelled compositions). That requires actual
structure data (atomic positions + space-group symmetries) which the
`scl.candidates.Candidate` representation doesn't currently model — so the
GNN upgrade is gated on first extending the candidate space, then training.
The MLP is the placeholder that proves the rest of the architecture
(loop, acquisitions, manifold curvature, NNQS proxy, falsification) takes
mean+std from the surrogate without caring whether it's a GP or an NN.

Lazy-imports torch — importing this module is cheap and works without
torch; instantiating ``TorchSurrogate`` requires it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


class TorchSurrogate:
    """MLP with MC dropout. Same interface as ``scl.neural.GPSurrogate``."""

    def __init__(
        self,
        hidden: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
        lr: float = 1e-2,
        n_epochs: int = 300,
        mc_samples: int = 32,
        seed: int = 0,
    ):
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "torch is required for TorchSurrogate. "
                "Install with: pip install -e '.[gnn]'"
            ) from e

        self.hidden = hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.n_epochs = n_epochs
        self.mc_samples = mc_samples
        self.seed = seed

        # Public attributes that mirror GPSurrogate so loop/manifold/diffphys
        # can use TorchSurrogate without modification.
        self.X_train: Optional[np.ndarray] = None
        self.x_mean: Optional[np.ndarray] = None
        self.x_std: Optional[np.ndarray] = None
        self.y_mean: float = 0.0
        self.y_std: float = 1.0
        self._net = None

    def _build(self, d_in: int):
        import torch.nn as nn

        layers: list = []
        for i in range(self.n_layers):
            d_prev = d_in if i == 0 else self.hidden
            layers.extend([
                nn.Linear(d_prev, self.hidden),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ])
        layers.append(nn.Linear(self.hidden, 1))
        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        import torch

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y row counts disagree")

        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0)
        self.x_std = np.where(self.x_std < 1e-9, 1.0, self.x_std)
        Xs = (X - self.x_mean) / self.x_std

        self.y_mean = float(y.mean())
        y_std = float(y.std())
        self.y_std = y_std if y_std > 1e-9 else 1.0
        ys = (y - self.y_mean) / self.y_std

        torch.manual_seed(self.seed)
        Xt = torch.from_numpy(Xs.astype(np.float32))
        yt = torch.from_numpy(ys.astype(np.float32)).unsqueeze(1)

        self._net = self._build(X.shape[1])
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        self._net.train()
        for _ in range(self.n_epochs):
            opt.zero_grad()
            pred = self._net(Xt)
            loss = loss_fn(pred, yt)
            loss.backward()
            opt.step()

        self.X_train = Xs

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]

        if self._net is None or self.X_train is None:
            n = X.shape[0]
            return (
                np.full(n, self.y_mean, dtype=np.float64),
                np.full(n, self.y_std, dtype=np.float64),
            )

        Xs = (X - self.x_mean) / self.x_std
        Xt = torch.from_numpy(Xs.astype(np.float32))

        # MC dropout: keep the net in train() so dropout stays on, then
        # average / std-dev across `mc_samples` forward passes.
        self._net.train()
        with torch.no_grad():
            samples = torch.stack(
                [self._net(Xt) for _ in range(self.mc_samples)]
            )
        # samples shape: (mc_samples, n_test, 1)
        mean_norm = samples.mean(dim=0).squeeze(-1).numpy()
        std_norm = samples.std(dim=0).squeeze(-1).numpy()
        # De-standardize back to Tc units.
        return (
            mean_norm.astype(np.float64) * self.y_std + self.y_mean,
            np.maximum(std_norm.astype(np.float64) * self.y_std, 1e-3),
        )


def make_surrogate(kind: str = "gp", **kwargs):
    """Factory: build a surrogate by name.

    ``kind="gp"`` returns a :class:`scl.neural.GPSurrogate` (default, no
    extra deps). ``kind="nn"`` returns a :class:`TorchSurrogate` (requires
    the ``[gnn]`` extra). Any unknown kind raises ``ValueError``.
    """
    if kind == "gp":
        from .neural import GPSurrogate
        return GPSurrogate(**kwargs)
    if kind == "nn":
        return TorchSurrogate(**kwargs)
    raise ValueError(f"unknown surrogate kind: {kind!r}")
