"""Neural-network quantum state (RBM ansatz) for the transverse-field Ising chain.

Stand-in for full DFT / quantum Monte Carlo in the discovery loop. Given an
'effective Hamiltonian' (J, h) derived from a candidate's features, NNQS
estimates the variational ground-state energy of an RBM wavefunction over the
full 2^N basis (N is small enough to enumerate exactly).

The log-wavefunction is the canonical Carleo-Troyer ansatz:

    log psi(s) = a . s + sum_h log cosh(b_h + W_h . s)

with analytic gradients via the standard variational Monte Carlo formula

    dE/dtheta = 2 * (<E_loc * O_theta> - <E_loc> * <O_theta>)
"""

from __future__ import annotations

import numpy as np

from .candidates import Candidate, featurize


class RBMWavefunction:
    def __init__(self, n_sites: int = 6, n_hidden: int = 8, seed: int = 0):
        if n_sites > 12:
            raise ValueError("full-enumeration RBM only supports n_sites <= 12")
        self.N = n_sites
        self.M = n_hidden
        rng = np.random.default_rng(seed)
        self.a = rng.normal(0.0, 0.05, size=n_sites)
        self.b = rng.normal(0.0, 0.05, size=n_hidden)
        self.W = rng.normal(0.0, 0.05, size=(n_hidden, n_sites))
        self.states = self._enumerate()

    def _enumerate(self) -> np.ndarray:
        return np.array(
            [
                [(-1.0 if (i >> k) & 1 else 1.0) for k in range(self.N)]
                for i in range(2 ** self.N)
            ],
            dtype=float,
        )

    def log_psi(self, S: np.ndarray) -> np.ndarray:
        z = self.b[None, :] + S @ self.W.T
        return S @ self.a + np.sum(np.log(np.cosh(z)), axis=1)

    def _local_energy(self, J: float, h: float) -> tuple[np.ndarray, np.ndarray]:
        S = self.states
        log_psi = self.log_psi(S)
        # Diagonal: -J Σ s_i s_{i+1} (periodic boundary)
        sz_sz = np.sum(S * np.roll(S, -1, axis=1), axis=1)
        E_loc = -J * sz_sz.astype(float)
        # Off-diagonal: -h Σ σx_i — flips one spin
        for i in range(self.N):
            S_flip = S.copy()
            S_flip[:, i] *= -1.0
            ratio = np.exp(self.log_psi(S_flip) - log_psi)
            E_loc += -h * ratio
        return E_loc, log_psi

    def energy(self, J: float, h: float) -> float:
        E_loc, log_psi = self._local_energy(J, h)
        log_psi -= log_psi.max()
        psi2 = np.exp(2.0 * log_psi)
        return float((psi2 * E_loc).sum() / psi2.sum())

    def gradient(self, J: float, h: float) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        S = self.states
        E_loc, log_psi = self._local_energy(J, h)
        log_psi -= log_psi.max()
        psi2 = np.exp(2.0 * log_psi)
        Z = psi2.sum()
        p = psi2 / Z

        z_arg = self.b[None, :] + S @ self.W.T
        tanh_z = np.tanh(z_arg)

        # O_theta(s) for each parameter family.
        O_a = S                                    # (n_states, N)
        O_b = tanh_z                               # (n_states, M)
        O_W = tanh_z[:, :, None] * S[:, None, :]   # (n_states, M, N)

        E = float((p * E_loc).sum())
        pE = p * E_loc
        mean_O_a = (p[:, None] * O_a).sum(axis=0)
        mean_O_b = (p[:, None] * O_b).sum(axis=0)
        mean_O_W = (p[:, None, None] * O_W).sum(axis=0)

        grad_a = 2.0 * ((pE[:, None] * O_a).sum(axis=0) - E * mean_O_a)
        grad_b = 2.0 * ((pE[:, None] * O_b).sum(axis=0) - E * mean_O_b)
        grad_W = 2.0 * ((pE[:, None, None] * O_W).sum(axis=0) - E * mean_O_W)
        return E, (grad_a, grad_b, grad_W)

    def fit(self, J: float, h: float, steps: int = 120, lr: float = 0.05) -> list[float]:
        history: list[float] = []
        for _ in range(steps):
            E, (ga, gb, gW) = self.gradient(J, h)
            self.a -= lr * ga
            self.b -= lr * gb
            self.W -= lr * gW
            history.append(E)
        return history


def exact_ground_energy(n_sites: int, J: float, h: float) -> float:
    """Exact diagonalization of TFIM with periodic boundary, for verification."""
    n = n_sites
    dim = 2 ** n
    H = np.zeros((dim, dim))
    for s in range(dim):
        # Diagonal Σz Σz term
        zz = 0
        for i in range(n):
            si = 1 - 2 * ((s >> i) & 1)
            sj = 1 - 2 * ((s >> ((i + 1) % n)) & 1)
            zz += si * sj
        H[s, s] = -J * zz
        # Off-diagonal Σ σx_i flips bit i
        for i in range(n):
            sp = s ^ (1 << i)
            H[s, sp] += -h
    eigvals = np.linalg.eigvalsh(H)
    return float(eigvals[0])


def quantum_proxy(c: Candidate, n_sites: int = 6, n_hidden: int = 8,
                  steps: int = 80, lr: float = 0.05) -> float:
    """Map a candidate to (J, h) and return per-site variational ground energy.

    Heuristic mapping:
      J — exchange-like coupling, scaled by EN contrast (covalent vs ionic mix).
      h — transverse field, scaled by H content (light-atom kinetic energy).

    Returns energy per site; lower (more negative) = stronger coupling regime,
    correlating positively with phonon-mediated superconductivity.
    """
    f = featurize(c)
    en_diff, h_frac = float(f[5]), float(f[4])
    J = max(0.1, 1.0 + 0.5 * (en_diff - 1.5))
    h_field = 1.0 + 2.0 * h_frac
    rbm = RBMWavefunction(n_sites=n_sites, n_hidden=n_hidden,
                          seed=hash(c.formula()) & 0xFFFF)
    rbm.fit(J, h_field, steps=steps, lr=lr)
    return rbm.energy(J, h_field) / n_sites
