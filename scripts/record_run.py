#!/usr/bin/env python
"""Record a closed-loop discovery run + paired baseline as JSONL.

Output goes to ``docs/recorded/{active,baseline}.jsonl`` so the static page
under ``docs/recorded/index.html`` can fetch and render it without needing
the FastAPI backend.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make `scl` importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scl.loop import run_loop  # noqa: E402
from scl.web.runner import serialize_round, serialize_summary  # noqa: E402


def _dump(path: Path, label: str, config: dict, result, elapsed: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(json.dumps({
            "type": "meta",
            "label": label,
            "config": config,
            "elapsed_seconds": elapsed,
        }) + "\n")
        for rl in result.rounds:
            f.write(json.dumps(serialize_round(rl)) + "\n")
        f.write(json.dumps({"type": "summary", **serialize_summary(result)}) + "\n")
    print(f"  wrote {path} ({len(result.rounds)} rounds)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="docs/recorded",
                        help="output directory for the JSONL files")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-size", type=int, default=200)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    common = dict(
        rounds=args.rounds,
        seed=args.seed,
        pool_size=args.pool_size,
        init_size=5,
        kappa=2.0,
        falsify_every=5,
        inverse_every=7,
        nnqs_every=6,
        manifold_weight=0.5,
        target_tc_k=320.0,
    )

    print("running active loop ...")
    t0 = time.time()
    active = run_loop(**common)
    active_elapsed = time.time() - t0
    print(f"  best Tc: {active.best_tc_k:.1f} K  ({active_elapsed:.1f}s)")

    print("running baseline (random) ...")
    t0 = time.time()
    baseline = run_loop(
        **{**common, "random_select_only": True, "falsify_every": 0,
           "inverse_every": 0, "nnqs_every": 0, "manifold_weight": 0.0},
    )
    baseline_elapsed = time.time() - t0
    print(f"  best Tc: {baseline.best_tc_k:.1f} K  ({baseline_elapsed:.1f}s)")

    _dump(out_dir / "active.jsonl", "active learning", common, active, active_elapsed)
    _dump(
        out_dir / "baseline.jsonl",
        "random baseline",
        {**common, "random_select_only": True},
        baseline,
        baseline_elapsed,
    )

    print(f"\n  Δ best Tc (active − baseline): {active.best_tc_k - baseline.best_tc_k:+.1f} K")
    return 0


if __name__ == "__main__":
    sys.exit(main())
