"""Closed-loop discovery orchestrator.

Each round:
  1. Sample a candidate pool.
  2. Symbolic veto removes physics-violating candidates.
  3. Surrogate scores survivors with mean + uncertainty.
  4. UCB selection (with optional manifold-curvature bonus), or random baseline.
  5. Periodic overrides:
       * `falsify_every` rounds → adversarial probe of current best.
       * `inverse_every` rounds → diffphys inverse-design proposal.
  6. NNQS quantum-proxy second opinion on the leading candidate (every
     `nnqs_every` rounds).
  7. Mock lab synthesizes + measures (process layer may drift the phase).
  8. Successful measurements feed back into the surrogate, indexed by the
     *realized* candidate (the loop learns 'makeable', not just 'good').
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .active import ei_select, random_select, thompson_select, ucb_select
from .candidates import Candidate, featurize, sample_random
from .diffphys import inverse_design
from .falsify import falsify_neighbors
from .lab import Lab
from .manifold import manifold_bonus
from .neural import GPSurrogate
from .nnqs import quantum_proxy
from .symbolic import symbolic_check


@dataclass
class RoundLog:
    round: int
    candidate: Candidate
    realized: Candidate
    predicted_mean: Optional[float]
    predicted_std: Optional[float]
    quantum_proxy: Optional[float]
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


def _ucb_with_manifold(
    survivors: list[Candidate],
    model: GPSurrogate,
    kappa: float,
    manifold_weight: float,
) -> tuple[Candidate, float, float]:
    feats = np.stack([featurize(c) for c in survivors])
    mu, sd = model.predict(feats)
    score = mu + kappa * sd
    if manifold_weight > 0.0:
        # Curvature is expensive; only apply to the top-N pre-screen.
        top = np.argsort(-score)[: min(20, len(survivors))]
        for i in top:
            score[i] += manifold_bonus(survivors[int(i)], model, manifold_weight)
    j = int(np.argmax(score))
    return survivors[j], float(mu[j]), float(sd[j])


def run_loop(
    rounds: int = 30,
    seed: int = 42,
    pool_size: int = 200,
    init_size: int = 5,
    kappa: float = 2.0,
    falsify_every: int = 5,
    inverse_every: int = 7,
    nnqs_every: int = 6,
    manifold_weight: float = 0.5,
    target_tc_k: float = 320.0,
    random_select_only: bool = False,
    acquisition: str = "ucb",
    world_mode: str = "single",
    use_agent: bool = False,
    agent_model: str = "claude-opus-4-7",
    agent_effort: str = "xhigh",
    verbose: bool = False,
    on_round: Optional[Callable[["RoundLog"], None]] = None,
) -> LoopResult:
    rng = np.random.default_rng(seed)
    lab = Lab(rng=rng, world_mode=world_mode)
    model = GPSurrogate()

    agent = None
    if use_agent:
        from .agent import AgentTools, LLMHypothesizer
        agent = LLMHypothesizer(model=agent_model, effort=agent_effort)

    X_train: list[np.ndarray] = []
    y_train: list[float] = []
    seen: list[Candidate] = []
    result = LoopResult()

    # Cold start.
    for c in _seed_initial(init_size, rng):
        m = lab.run(c)
        if m.success:
            X_train.append(featurize(m.candidate))
            y_train.append(m.tc_k)
            seen.append(m.candidate)
        result.rounds.append(
            RoundLog(
                round=-1,
                candidate=c,
                realized=m.candidate,
                predicted_mean=None,
                predicted_std=None,
                quantum_proxy=None,
                measured_tc_k=m.tc_k,
                success=m.success,
                note="seed " + (m.note or ""),
                best_so_far_k=max(y_train) if y_train else 0.0,
            )
        )

    if on_round is not None:
        for rl in result.rounds:
            on_round(rl)

    if X_train:
        model.fit(np.stack(X_train), np.array(y_train))

    for r in range(rounds):
        pool = [sample_random(rng) for _ in range(pool_size)]
        survivors = [c for c in pool if symbolic_check(c).ok]
        if not survivors:
            continue

        chosen: Candidate
        pred_mean: Optional[float] = None
        pred_std: Optional[float] = None
        nnqs_e: Optional[float] = None
        note = ""

        if agent is not None:
            from .agent import AgentTools
            tools = AgentTools(
                model=model,
                seen=seen,
                y_train=y_train,
                rng=rng,
                target_tc_k=target_tc_k,
            )
            try:
                proposal = agent.propose(tools, round_idx=r, total_rounds=rounds)
            except Exception as e:
                proposal = None
                note = f"agent error ({type(e).__name__}); fell back to UCB"
            if proposal is not None and symbolic_check(proposal).ok:
                chosen = proposal
                if X_train:
                    m_, s_ = model.predict(featurize(chosen))
                    pred_mean, pred_std = float(m_[0]), float(s_[0])
                note = f"agent ({agent.model})"
                meas = lab.run(chosen)
                if meas.success:
                    X_train.append(featurize(meas.candidate))
                    y_train.append(meas.tc_k)
                    seen.append(meas.candidate)
                    model.fit(np.stack(X_train), np.array(y_train))
                best_so_far = max(y_train) if y_train else 0.0
                round_log = RoundLog(
                    round=r,
                    candidate=chosen,
                    realized=meas.candidate,
                    predicted_mean=pred_mean,
                    predicted_std=pred_std,
                    quantum_proxy=nnqs_e,
                    measured_tc_k=meas.tc_k,
                    success=meas.success,
                    note=note + (f" [{meas.note}]" if meas.note else ""),
                    best_so_far_k=best_so_far,
                )
                result.rounds.append(round_log)
                if on_round is not None:
                    on_round(round_log)
                if verbose:
                    tag = "OK" if meas.success else "FAIL"
                    tc = f"{meas.tc_k:6.1f}K" if meas.success else "  --   "
                    pred = (
                        f"{pred_mean:6.1f}±{pred_std:5.1f}"
                        if pred_mean is not None else "   n/a   "
                    )
                    print(
                        f"r{r:03d} {tag} pred={pred} measured={tc} "
                        f"best={best_so_far:6.1f}K  "
                        f"{meas.candidate.formula()} @ {meas.candidate.pressure_gpa:.0f}GPa "
                        f"({note})"
                    )
                continue

        do_falsify = (
            falsify_every > 0
            and r > 0
            and r % falsify_every == 0
            and y_train
        )
        do_inverse = (
            inverse_every > 0
            and r > 0
            and r % inverse_every == 0
            and y_train
            and not do_falsify
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
                do_falsify = False

        if not do_falsify and do_inverse:
            inv = inverse_design(target_tc_k, model, rng)
            if inv is not None:
                chosen = inv
                m_, s_ = model.predict(featurize(inv))
                pred_mean, pred_std = float(m_[0]), float(s_[0])
                note = f"inverse-design probe (target {target_tc_k:.0f}K)"
            else:
                do_inverse = False

        if not (do_falsify or do_inverse):
            if random_select_only or not X_train:
                chosen = random_select(survivors, rng, k=1)[0]
                if X_train:
                    m_, s_ = model.predict(featurize(chosen))
                    pred_mean, pred_std = float(m_[0]), float(s_[0])
                note = "random" if random_select_only else "cold-start random"
            elif acquisition == "ucb":
                chosen, pred_mean, pred_std = _ucb_with_manifold(
                    survivors, model, kappa, manifold_weight
                )
                note = "UCB+manifold" if manifold_weight > 0 else "UCB"
            elif acquisition == "ei":
                picks, mu, sd, _ = ei_select(
                    survivors, model, current_best=float(max(y_train)), k=1,
                )
                chosen = picks[0]
                pred_mean, pred_std = float(mu[0]), float(sd[0])
                note = "EI"
            elif acquisition == "thompson":
                picks, mu, sd, _ = thompson_select(survivors, model, rng, k=1)
                chosen = picks[0]
                pred_mean, pred_std = float(mu[0]), float(sd[0])
                note = "Thompson"
            else:
                raise ValueError(f"unknown acquisition: {acquisition!r}")

        # NNQS second opinion: catch surrogate hallucinations on top picks.
        if nnqs_every > 0 and r > 0 and r % nnqs_every == 0 and pred_mean is not None:
            nnqs_e = quantum_proxy(chosen, n_sites=6, n_hidden=6, steps=40, lr=0.05)
            note += f" (NNQS E/site={nnqs_e:+.3f})"

        meas = lab.run(chosen)

        if meas.success:
            X_train.append(featurize(meas.candidate))
            y_train.append(meas.tc_k)
            seen.append(meas.candidate)
            model.fit(np.stack(X_train), np.array(y_train))

        best_so_far = max(y_train) if y_train else 0.0
        round_log = RoundLog(
            round=r,
            candidate=chosen,
            realized=meas.candidate,
            predicted_mean=pred_mean,
            predicted_std=pred_std,
            quantum_proxy=nnqs_e,
            measured_tc_k=meas.tc_k,
            success=meas.success,
            note=note + (f" [{meas.note}]" if meas.note else ""),
            best_so_far_k=best_so_far,
        )
        result.rounds.append(round_log)
        if on_round is not None:
            on_round(round_log)

        if verbose:
            tag = "OK" if meas.success else "FAIL"
            tc = f"{meas.tc_k:6.1f}K" if meas.success else "  --   "
            pred = (
                f"{pred_mean:6.1f}±{pred_std:5.1f}"
                if pred_mean is not None else "   n/a   "
            )
            print(
                f"r{r:03d} {tag} pred={pred} measured={tc} "
                f"best={best_so_far:6.1f}K  "
                f"{meas.candidate.formula()} @ {meas.candidate.pressure_gpa:.0f}GPa "
                f"({note})"
            )

    if y_train:
        i = int(np.argmax(y_train))
        result.best_candidate = seen[i]
        result.best_tc_k = float(y_train[i])

    return result
