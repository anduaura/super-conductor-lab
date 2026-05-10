"""CLI entry point: `scl run [...]`."""

from __future__ import annotations

import argparse
import sys

from .loop import LoopResult, run_loop


def _print_summary(label: str, result: LoopResult) -> None:
    successes = [r for r in result.rounds if r.success]
    print(f"\n=== {label} ===")
    print(f"rounds            : {len(result.rounds)}")
    print(f"successful runs   : {len(successes)} / {len(result.rounds)}")
    if result.best_candidate is not None:
        print(f"best Tc           : {result.best_tc_k:.1f} K")
        print(f"best composition  : {result.best_candidate.formula()}")
        print(f"best pressure     : {result.best_candidate.pressure_gpa:.0f} GPa")
    else:
        print("best              : (no successful syntheses)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="scl",
        description="Closed-loop superconductor discovery (simulator).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Launch the web UI.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--runs-dir", default="runs")
    serve.add_argument("--auth-token", default=None,
                       help="if set (or env SCL_AUTH_TOKEN), require Bearer auth on /api/*")

    run = sub.add_parser("run", help="Run the closed-loop discovery engine.")
    run.add_argument("--rounds", type=int, default=30)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--pool", type=int, default=200, dest="pool_size")
    run.add_argument("--init", type=int, default=5, dest="init_size")
    run.add_argument("--kappa", type=float, default=2.0)
    run.add_argument("--falsify-every", type=int, default=5)
    run.add_argument("--inverse-every", type=int, default=7)
    run.add_argument("--nnqs-every", type=int, default=6)
    run.add_argument("--manifold-weight", type=float, default=0.5)
    run.add_argument("--target-tc", type=float, default=320.0)
    run.add_argument("--acquisition", default="ucb",
                     choices=["ucb", "ei", "thompson"])
    run.add_argument("--world-mode", default="single",
                     choices=["single", "multi", "ambient"],
                     help="ground-truth Tc landscape (single peak vs multi-modal)")
    run.add_argument("--use-agent", action="store_true",
                     help="drive selection with an LLM hypothesizer (requires [agent] extras)")
    run.add_argument("--agent-model", default="claude-opus-4-7")
    run.add_argument("--agent-effort", default="xhigh",
                     choices=["low", "medium", "high", "xhigh", "max"])
    run.add_argument("--baseline", action="store_true",
                     help="also run an equivalent random-search baseline")
    run.add_argument("--quiet", action="store_true")

    bench = sub.add_parser("bench", help="Strategy × seed benchmark grid.")
    bench.add_argument("--strategies",
                       default="random,ucb,ei,thompson,ucb+manifold,ucb+falsify,ucb+inverse,all",
                       help="comma-separated strategy names")
    bench.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10",
                       help="comma-separated integer seeds")
    bench.add_argument("--rounds", type=int, default=30)
    bench.add_argument("--pool", type=int, default=200, dest="pool_size")
    bench.add_argument("--init", type=int, default=5, dest="init_size")
    bench.add_argument("--world-mode", default="single",
                       choices=["single", "multi", "ambient"])
    bench.add_argument("--out", default="bench.csv",
                       help="output CSV path")

    args = p.parse_args(argv)

    if args.cmd == "serve":
        try:
            import uvicorn
            from .web.app import create_app
        except ImportError as e:
            print(f"web extras not installed: {e}\n"
                  f"hint: pip install -e '.[web]'", file=sys.stderr)
            return 2
        app = create_app(runs_dir=args.runs_dir, auth_token=args.auth_token)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        return 0

    if args.cmd == "run":
        active = run_loop(
            rounds=args.rounds,
            seed=args.seed,
            pool_size=args.pool_size,
            init_size=args.init_size,
            kappa=args.kappa,
            falsify_every=args.falsify_every,
            inverse_every=args.inverse_every,
            nnqs_every=args.nnqs_every,
            manifold_weight=args.manifold_weight,
            target_tc_k=args.target_tc,
            acquisition=args.acquisition,
            world_mode=args.world_mode,
            use_agent=args.use_agent,
            agent_model=args.agent_model,
            agent_effort=args.agent_effort,
            verbose=not args.quiet,
        )
        _print_summary("active learning" + (" (LLM agent)" if args.use_agent else ""), active)

        if args.baseline:
            baseline = run_loop(
                rounds=args.rounds,
                seed=args.seed,
                pool_size=args.pool_size,
                init_size=args.init_size,
                kappa=args.kappa,
                falsify_every=0,
                inverse_every=0,
                nnqs_every=0,
                manifold_weight=0.0,
                world_mode=args.world_mode,
                random_select_only=True,
                verbose=False,
            )
            _print_summary("random baseline", baseline)

    if args.cmd == "bench":
        from .bench import format_summary, run_grid, summarize, to_csv
        strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        print(f"benchmarking {len(strategies)} strategies × {len(seeds)} seeds × "
              f"{args.rounds} rounds (world_mode={args.world_mode})")
        rows = run_grid(
            strategies, seeds, args.rounds,
            world_mode=args.world_mode,
            pool_size=args.pool_size,
            init_size=args.init_size,
            progress=True,
        )
        to_csv(rows, args.out)
        print(f"\nwrote {args.out} ({len(rows)} rows)\n")
        print(format_summary(summarize(rows)))

    return 0


if __name__ == "__main__":
    sys.exit(main())
