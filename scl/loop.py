"""Closed-loop discovery orchestrator.

Each round:
  1. Sample a candidate pool.
  2. Symbolic veto removes physics-violating candidates.
  3. Surrogate scores survivors with mean + uncertainty.
  4. Either UCB selection (default) or random selection (baseline).
  5. Every `falsify_every` rounds, override step 4 with a falsification probe
     of the current best.
  6. Mock lab synthesizes + measures.
  7. Successful measurements feed back into the surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .active import random_select, ucb_select
from .candidates import Candidate, featurize, sample_random
from .falsify import falsify_neighbors
from .lab import Lab, MeasurementResult
from .neural import GPSurrogate
from .symbolic import symbolic_check


@dataclass
class RoundLog:
    round: int
    candidate: Candidate
    predicted_mean: Optional[float]
    predicted_std: Optional[float]
    measured_tc_k: Optional[float]
    success: bool
    note: str
    best_so_far_k: float


@dataclass
class LoopResult:
    rounds: list[RoundLog] = field(default_factory=list)
    best_candidate: Optional[Candidate] = None
    best_tc_k: float = 0.0

    def measured(self) -> list[float]:
        return [r.measured_tc_k for r in self.rounds if r.success]


def _seed_initial(
    n: int, rng: np.random.Generator, max_tries: int = 1000
) -> list[Candidate]:
    out: list[Candidate] = []
    for _ in range(max_tries):
        c = sample_random(rng)
        if symbolic_check(c).ok:
            out.append(c)
            if len(out) >= n:
                return out
    return out


def run_loop(
    rounds: int = 30,
    seed: int = 42,
    pool_size: int = 200,
    init_size: int = 5,
    kappa: float = 2.0,
    falsify_every: int = 5,
    random_select_only: bool = False,
    verbose: bool = False,
) -> LoopResult:
    rng = np.random.default_rng(seed)
    lab = Lab(rng=rng)
    model = GPSurrogate()

    X_train: list[np.ndarray] = []
    y_train: list[float] = []
    seen: list[Candidate] = []
    result = LoopResult()

    # Cold start.
    for c in _seed_initial(init_size, rng):
        m = lab.run(c)
        if m.success:
            X_train.append(featurize(c))
            y_train.append(m.tc_k)
            seen.append(c)
        result.rounds.append(
            RoundLog(
                round=-1,
                candidate=c,
                predicted_mean=None,
                predicted_std=None,
                measured_tc_k=m.tc_k,
                success=m.success,
                note="seed " + (m.note or ""),
                best_so_far_k=max(y_train) if y_train else 0.0,
            )
        )

    if X_train:
        model.fit(np.stack(X_train), np.array(y_train))

    for r in range(rounds):
        # 1. pool
        pool = [sample_random(rng) for _ in range(pool_size)]
        # 2. veto
        survivors = [c for c in pool if symbolic_check(c).ok]
        if not survivors:
            continue

        chosen: Candidate
        pred_mean: Optional[float] = None
        pred_std: Optional[float] = None
        note = ""

        # 5. falsification override
        do_falsify = (
            falsify_every > 0
            and r > 0
            and r % falsify_every == 0
            and y_train
        )

        if do_falsify:
            best_idx = int(np.argmax(y_train))
            probe = falsify_neighbors(seen[best_idx], model, rng)
            if probe is not None:
                chosen = probe
                m_, s_ = model.predict(featurize(probe))
                pred_mean, pred_std = float(m_[0]), float(s_[0])
                note = "falsification probe of current best"
            else:
                do_falsify = False  # fall through

        if not do_falsify:
            if random_select_only or not X_train:
                chosen = random_select(survivors, rng, k=1)[0]
                if X_train:
                    m_, s_ = model.predict(featurize(chosen))
                    pred_mean, pred_std = float(m_[0]), float(s_[0])
                note = "random" if random_select_only else "cold-start random"
            else:
                picks, mu, sd, _ = ucb_select(survivors, model, kappa=kappa, k=1)
                chosen = picks[0]
                pred_mean, pred_std = float(mu[0]), float(sd[0])
                note = "UCB"

        # 6. lab
        meas = lab.run(chosen)

        # 7. update
        if meas.success:
            X_train.append(featurize(chosen))
            y_train.append(meas.tc_k)
            seen.append(chosen)
            model.fit(np.stack(X_train), np.array(y_train))

        best_so_far = max(y_train) if y_train else 0.0
        result.rounds.append(
            RoundLog(
                round=r,
                candidate=chosen,
                predicted_mean=pred_mean,
                predicted_std=pred_std,
                measured_tc_k=meas.tc_k,
                success=meas.success,
                note=note + (f" [{meas.note}]" if meas.note else ""),
                best_so_far_k=best_so_far,
            )
        )

        if verbose:
            tag = "OK" if meas.success else "FAIL"
            tc = f"{meas.tc_k:6.1f}K" if meas.success else "  --   "
            pred = f"{pred_mean:6.1f}±{pred_std:5.1f}" if pred_mean is not None else "   n/a   "
            print(
                f"r{r:03d} {tag} pred={pred} measured={tc} "
                f"best={best_so_far:6.1f}K  {chosen.formula()} @ {chosen.pressure_gpa:.0f}GPa"
                f"  ({note})"
            )

    if y_train:
        i = int(np.argmax(y_train))
        result.best_candidate = seen[i]
        result.best_tc_k = float(y_train[i])

    return result
