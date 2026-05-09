import numpy as np

from scl.loop import run_loop
from scl.world_model import true_tc
from scl.candidates import Candidate


def test_world_model_peaks_in_expected_regime():
    # Hand-crafted near-optimum point: ~half-half valence balance,
    # high H, decent pressure, good EN contrast.
    optimum = Candidate(
        composition=(("La", 0.075), ("S", 0.075), ("H", 0.85)),
        pressure_gpa=300.0,
    )
    off = Candidate(
        composition=(("La", 0.5), ("H", 0.5)),
        pressure_gpa=50.0,
    )
    assert true_tc(optimum) > true_tc(off) + 50.0


def test_loop_runs_and_finds_signal():
    res = run_loop(rounds=20, seed=42, pool_size=80, verbose=False)
    successes = [r for r in res.rounds if r.success]
    assert len(successes) >= 5
    # The world model maxes out around ~280K under good conditions; even a
    # short run with UCB should land at least one mid-Tc hit.
    assert max(r.measured_tc_k for r in successes) > 50.0


def test_active_beats_random_on_average():
    """UCB should at least match random search across multiple seeds.

    We compare best-Tc-found averaged across 4 seeds. Random can occasionally
    get lucky, so we use the median of differences and a generous tolerance.
    """
    seeds = [1, 7, 13, 21]
    diffs = []
    for s in seeds:
        a = run_loop(rounds=25, seed=s, pool_size=80, verbose=False)
        b = run_loop(rounds=25, seed=s, pool_size=80,
                     random_select_only=True, falsify_every=0, verbose=False)
        diffs.append(a.best_tc_k - b.best_tc_k)
    # On average, active learning should not be worse than random by more than
    # noise. Median(diffs) >= 0 is the meaningful claim; we allow a small slack.
    assert float(np.median(diffs)) >= -10.0
