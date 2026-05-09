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


def test_run_one_multi_mode_runs():
    row = run_one("ucb", seed=0, rounds=4, world_mode="multi",
                  pool_size=20, init_size=2)
    assert row.world_mode == "multi"
    assert row.best_tc_k >= 0
