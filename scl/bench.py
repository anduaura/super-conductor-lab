"""Benchmarking harness — strategy × seed grid against the closed loop.

Each strategy is a named bundle of ``run_loop`` kwargs; the harness sweeps
seeds, records best Tc + summary stats, and writes everything to CSV. The
CLI exposes this as ``scl bench``.

The pillar-ablation strategies (``ucb+manifold``, ``ucb+falsify``,
``ucb+inverse``, ``all``) let us isolate which architectural components are
actually pulling weight on a given landscape.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .loop import run_loop


_BASE_KW = dict(
    falsify_every=0,
    inverse_every=0,
    nnqs_every=0,
    manifold_weight=0.0,
    random_select_only=False,
    acquisition="ucb",
    use_agent=False,
)


_STRATEGIES: dict[str, dict] = {
    "random": {**_BASE_KW, "random_select_only": True},
    "ucb": {**_BASE_KW, "acquisition": "ucb"},
    "ei": {**_BASE_KW, "acquisition": "ei"},
    "thompson": {**_BASE_KW, "acquisition": "thompson"},
    "ucb+manifold": {**_BASE_KW, "acquisition": "ucb", "manifold_weight": 0.5},
    "ucb+falsify": {**_BASE_KW, "acquisition": "ucb", "falsify_every": 5},
    "ucb+inverse": {**_BASE_KW, "acquisition": "ucb", "inverse_every": 7},
    "ucb+anneal": {
        **_BASE_KW, "acquisition": "ucb",
        "kappa": 4.0, "kappa_end": 0.5,
    },
    "ucb+anneal+manifold": {
        **_BASE_KW, "acquisition": "ucb",
        "kappa": 4.0, "kappa_end": 0.5, "manifold_weight": 0.5,
    },
    "all": {
        **_BASE_KW,
        "acquisition": "ucb",
        "manifold_weight": 0.5,
        "falsify_every": 5,
        "inverse_every": 7,
        # NNQS is slow; off by default in bench. Enable explicitly via "all+nnqs".
    },
    "all+nnqs": {
        **_BASE_KW,
        "acquisition": "ucb",
        "manifold_weight": 0.5,
        "falsify_every": 5,
        "inverse_every": 7,
        "nnqs_every": 6,
    },
}


def available_strategies() -> list[str]:
    return list(_STRATEGIES.keys())


@dataclass
class BenchResult:
    strategy: str
    seed: int
    world_mode: str
    rounds: int
    successful_rounds: int
    best_tc_k: float
    best_formula: Optional[str]
    best_pressure_gpa: Optional[float]
    elapsed_seconds: float


def run_one(
    strategy: str,
    seed: int,
    rounds: int,
    world_mode: str = "single",
    pool_size: int = 200,
    init_size: int = 5,
) -> BenchResult:
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"unknown strategy {strategy!r}; available: {available_strategies()}"
        )
    kwargs = _STRATEGIES[strategy]
    t0 = time.time()
    result = run_loop(
        rounds=rounds,
        seed=seed,
        pool_size=pool_size,
        init_size=init_size,
        world_mode=world_mode,
        verbose=False,
        **kwargs,
    )
    elapsed = time.time() - t0
    successes = sum(1 for r in result.rounds if r.success)
    best = result.best_candidate
    return BenchResult(
        strategy=strategy,
        seed=seed,
        world_mode=world_mode,
        rounds=len(result.rounds),
        successful_rounds=successes,
        best_tc_k=result.best_tc_k,
        best_formula=best.formula() if best else None,
        best_pressure_gpa=float(best.pressure_gpa) if best else None,
        elapsed_seconds=elapsed,
    )


def run_grid(
    strategies: Iterable[str],
    seeds: Iterable[int],
    rounds: int,
    world_mode: str = "single",
    pool_size: int = 200,
    init_size: int = 5,
    progress: bool = False,
) -> list[BenchResult]:
    rows: list[BenchResult] = []
    for strategy in strategies:
        for seed in seeds:
            row = run_one(strategy, seed, rounds, world_mode, pool_size, init_size)
            rows.append(row)
            if progress:
                print(
                    f"  {strategy:>14} seed={seed:>3} → {row.best_tc_k:6.1f}K  "
                    f"({row.elapsed_seconds:5.1f}s)"
                )
    return rows


def to_csv(rows: list[BenchResult], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "strategy", "seed", "world_mode", "rounds", "successful_rounds",
            "best_tc_k", "best_formula", "best_pressure_gpa", "elapsed_seconds",
        ])
        for r in rows:
            w.writerow([
                r.strategy, r.seed, r.world_mode, r.rounds, r.successful_rounds,
                f"{r.best_tc_k:.4f}",
                r.best_formula or "",
                f"{r.best_pressure_gpa:.2f}" if r.best_pressure_gpa is not None else "",
                f"{r.elapsed_seconds:.4f}",
            ])


def summarize(rows: list[BenchResult], threshold_k: float = 293.0) -> list[dict]:
    """Per-strategy aggregate stats.

    ``threshold_k`` (default 293 K, the room-temperature bar) is the cutoff
    for the per-strategy ``success_rate`` — fraction of seeds that produced
    at least one measurement above it.
    """
    by_strat: dict[str, list[float]] = {}
    by_strat_t: dict[str, list[float]] = {}
    for r in rows:
        by_strat.setdefault(r.strategy, []).append(r.best_tc_k)
        by_strat_t.setdefault(r.strategy, []).append(r.elapsed_seconds)
    out = []
    for strat, tcs in by_strat.items():
        a = np.array(tcs)
        out.append({
            "strategy": strat,
            "n": int(len(a)),
            "median_tc_k": float(np.median(a)),
            "mean_tc_k": float(np.mean(a)),
            "p25_tc_k": float(np.percentile(a, 25)),
            "p75_tc_k": float(np.percentile(a, 75)),
            "best_tc_k": float(np.max(a)),
            "worst_tc_k": float(np.min(a)),
            "median_elapsed_s": float(np.median(by_strat_t[strat])),
            "success_rate": float(np.mean(a >= threshold_k)),
            "threshold_k": float(threshold_k),
        })
    out.sort(key=lambda r: (-r["success_rate"], -r["median_tc_k"]))
    return out


def format_summary(summary: list[dict]) -> str:
    lines = []
    threshold = summary[0]["threshold_k"] if summary else 293.0
    threshold_label = f"P(>{int(threshold)}K)"
    header = (
        f"{'strategy':>14}  {'n':>3}  {'median':>8}  "
        f"{'p25':>8}  {'p75':>8}  {'best':>8}  "
        f"{threshold_label:>10}  {'time':>6}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for s in summary:
        lines.append(
            f"{s['strategy']:>14}  {s['n']:>3}  "
            f"{s['median_tc_k']:>7.1f}K  {s['p25_tc_k']:>7.1f}K  "
            f"{s['p75_tc_k']:>7.1f}K  {s['best_tc_k']:>7.1f}K  "
            f"{s['success_rate'] * 100:>9.0f}%  "
            f"{s['median_elapsed_s']:>5.1f}s"
        )
    return "\n".join(lines)
