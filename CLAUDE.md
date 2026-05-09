# CLAUDE.md

Operating notes for Claude when working in this repo.

## Project

`super-conductor-lab` is a numpy-only simulator of a closed-loop neuro-symbolic
discovery engine for room-temperature superconductor candidates. It maps the
S-AGI manifesto to runnable code without depending on real DFT, GPUs, or a real
self-driving lab.

## Branch + commit policy

- Commit directly to `main`. No feature branches unless the user asks for one.
- Always push after committing: `git push -u origin main`.
- Do not open PRs unless the user explicitly asks for one.
- Never use `--no-verify`, `--force`, or amend already-pushed commits.

## Code rules

- Python 3.10+, numpy only. Do not add torch, scipy, sklearn, or pandas without
  asking — the whole point of the prototype is that it's a single readable loop.
- No new top-level docs (`*.md`) without an explicit ask. `README.md` and this
  file are the only ones.
- Default to no comments. Add one only when the *why* is non-obvious.
- Prefer editing existing modules over adding new ones.
- Keep public API in `scl/__init__.py` aligned with what the CLI and tests use.

## Test rules

- `pytest -q` must be green before pushing. Twelve tests today; keep that bar.
- Add a test alongside any non-trivial change (rule, surrogate, acquisition,
  loop step). Tests should be deterministic — always seed the RNG.
- For convergence-style tests, compare median across multiple seeds, not a
  single seed.

## Architecture map

| Concern                | Module               |
| ---------------------- | -------------------- |
| System 1 surrogate     | `scl/neural.py`      |
| System 2 veto          | `scl/symbolic.py`    |
| Hidden ground truth    | `scl/world_model.py` |
| Mock self-driving lab  | `scl/lab.py`         |
| Active learning (UCB)  | `scl/active.py`      |
| Falsification probes   | `scl/falsify.py`     |
| Closed-loop driver     | `scl/loop.py`        |
| CLI                    | `scl/cli.py`         |

The world model is **hidden from the surrogate** — the lab is the only data
channel. Do not import `world_model` from `neural.py` or `loop.py` outside of
the lab path.

## Planning notes (open threads)

- Current acquisition is plain UCB. Worth exploring expected improvement and
  Thompson sampling once we have a benchmark harness.
- Falsification currently picks the model's most-pessimistic neighbor of the
  best. Could sharpen by picking the maximum *predicted-vs-prior disagreement*
  instead.
- Symbolic rules are deliberately coarse. A real version would call out to
  pymatgen / matminer for stability + charge balance.
- The world-model landscape is single-peak. Add a multi-modal variant to
  stress-test exploration.
- No persistence yet — every CLI run starts cold. A simple JSONL log of the
  history would unblock cross-run comparison and offline analysis.
