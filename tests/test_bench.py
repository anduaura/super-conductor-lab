from pathlib import Path

import pytest

from scl.bench import (
    available_strategies,
    format_summary,
    run_grid,
    run_one,
    summarize,
    to_csv,
)


_TINY = dict(rounds=4, world_mode="single", pool_size=20, init_size=2)


def test_available_strategies_covers_core_set():
    s = available_strategies()
    for required in ("random", "ucb", "ei", "thompson", "all"):
        assert required in s, f"missing strategy {required!r}"


def test_run_one_random():
    row = run_one("random", seed=0, **_TINY)
    assert row.strategy == "random"
    assert row.world_mode == "single"
    assert row.rounds == _TINY["rounds"] + _TINY["init_size"]
    assert row.best_tc_k >= 0


def test_run_one_unknown_strategy():
    with pytest.raises(ValueError):
        run_one("does-not-exist", seed=0, **_TINY)


def test_run_grid_writes_csv(tmp_path: Path):
    rows = run_grid(
        ["random", "ucb"], seeds=[1, 2], **_TINY,
    )
    assert len(rows) == 4
    out = tmp_path / "results.csv"
    to_csv(rows, out)
    text = out.read_text()
    assert "strategy,seed" in text.splitlines()[0]
    assert text.count("\n") == len(rows) + 1  # header + rows


def test_summary_orders_by_median():
    rows = run_grid(["random", "ucb"], seeds=[1, 2], **_TINY)
    summary = summarize(rows)
    assert len(summary) == 2
    # Sorted descending by median.
    assert summary[0]["median_tc_k"] >= summary[1]["median_tc_k"]
    formatted = format_summary(summary)
    assert "strategy" in formatted and "median" in formatted


def test_summary_includes_success_rate():
    rows = run_grid(["random", "ucb"], seeds=[1, 2], **_TINY)
    summary = summarize(rows, threshold_k=10.0)
    assert all("success_rate" in s for s in summary)
    assert all(0.0 <= s["success_rate"] <= 1.0 for s in summary)


def test_anneal_strategies_registered():
    """Annealed-κ strategies must be available alongside the static ones."""
    s = available_strategies()
    assert "ucb+anneal" in s
    assert "ucb+anneal+manifold" in s


def test_run_one_anneal_runs():
    """Annealed-κ strategy runs end-to-end and produces a finite result."""
    row = run_one("ucb+anneal", seed=0, **_TINY)
    assert row.strategy == "ucb+anneal"
    assert row.best_tc_k >= 0


def test_run_one_multi_mode_runs():
    row = run_one("ucb", seed=0, rounds=4, world_mode="multi",
                  pool_size=20, init_size=2)
    assert row.world_mode == "multi"
    assert row.best_tc_k >= 0
