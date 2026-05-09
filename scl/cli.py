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
    run.add_argument("--use-agent", action="store_true",
                     help="drive selection with an LLM hypothesizer (requires [agent] extras)")
    run.add_argument("--agent-model", default="claude-opus-4-7")
    run.add_argument("--agent-effort", default="xhigh",
                     choices=["low", "medium", "high", "xhigh", "max"])
    run.add_argument("--baseline", action="store_true",
                     help="also run an equivalent random-search baseline")
    run.add_argument("--quiet", action="store_true")

    args = p.parse_args(argv)

    if args.cmd == "serve":
        try:
            import uvicorn
            from .web.app import create_app
        except ImportError as e:
            print(f"web extras not installed: {e}\n"
                  f"hint: pip install -e '.[web]'", file=sys.stderr)
            return 2
        app = create_app(runs_dir=args.runs_dir)
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
                random_select_only=True,
                verbose=False,
            )
            _print_summary("random baseline", baseline)

    return 0


if __name__ == "__main__":
    sys.exit(main())
