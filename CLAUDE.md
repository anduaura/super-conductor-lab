# CLAUDE.md

Operating notes for Claude when working in this repo.

## Project

`super-conductor-lab` is a numpy-only simulator of a closed-loop neuro-symbolic
discovery engine for room-temperature superconductor candidates. It maps the
S-AGI manifesto to runnable code without depending on real DFT, GPUs, or a real
self-driving lab.

## Branch + commit policy

- Commit directly to `main`. No feature branches unless the user asks for one.
- Always push after committing.
- Never use `--no-verify`, `--force`, or amend already-pushed commits.

### Push path (environment-specific)

Direct `git push origin main` returns **HTTP 403** in this environment — the
remote enforces branch protection. Until that changes, land changes via a
short-lived branch + PR + merge:

    git checkout -b <topic>
    git push -u origin <topic>
    # mcp__github__create_pull_request  (base: main, head: <topic>)
    # mcp__github__merge_pull_request   (merge_method: "merge")
    git checkout main && git fetch origin main && git reset --hard origin/main
    git branch -D <topic>

Do not open a PR for the user to review unless explicitly asked — these PRs
exist solely to bypass the push restriction and should be merged immediately.

### Commit authorship

All commits must be authored as the user, not Claude. Pass `--author` on every
commit instead of changing `git config` (per safety protocol):

    git commit --author="Andu <andu.ucsd@gmail.com>" -m "..."

If a commit has already been made with the wrong author and is **not yet
pushed**, fix it with:

    git commit --amend --author="Andu <andu.ucsd@gmail.com>" --no-edit

Never amend an author on a pushed commit.

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

| Concern                       | Module               |
| ----------------------------- | -------------------- |
| System 1 surrogate            | `scl/neural.py`      |
| System 2 veto + soft rules    | `scl/symbolic.py`    |
| Hidden ground truth (DFT)     | `scl/world_model.py` |
| Process layer (synth + drift) | `scl/process.py`     |
| Mock self-driving lab         | `scl/lab.py`         |
| Active learning (UCB)         | `scl/active.py`      |
| Manifold-curvature bonus      | `scl/manifold.py`    |
| Falsification probes          | `scl/falsify.py`     |
| NNQS quantum proxy (RBM/TFIM) | `scl/nnqs.py`        |
| Differentiable inverse design | `scl/diffphys.py`    |
| Closed-loop driver            | `scl/loop.py`        |
| CLI                           | `scl/cli.py`         |

The world model is **hidden from the surrogate** — the lab is the only data
channel. Do not import `world_model` from `neural.py`, `loop.py`, `manifold.py`,
`diffphys.py`, or `nnqs.py`. The lab and the tests are the only callers of
`true_tc`.

## Milestones

### Milestone 1 — closed-loop scaffold (done)
- Numpy-only neural surrogate (GP), symbolic veto, mock lab, UCB acquisition,
  falsification probe, end-to-end orchestrator + CLI.
- Beat random search by ~25K on seed 42 / 30 rounds.

### Milestone 2 — virtual-brain pillars (done)
- **NNQS** (`scl/nnqs.py`): Carleo–Troyer RBM wavefunction over the full 2^N
  Hilbert space of a small TFIM, with analytic VMC gradients. Used as a
  per-candidate "second opinion" every `nnqs_every` rounds.
- **Information-manifold engine** (`scl/manifold.py`): numerical Hessian of the
  surrogate's mean prediction provides a curvature-of-belief acquisition bonus.
- **Differentiable physics** (`scl/diffphys.py`): inverse-design proposer that
  gradient-descends in feature space toward a target Tc, projects to a discrete
  composition, and runs the symbolic verifier before submitting to the lab.
- **Symbolic verifier** extended with formation-driving-force and Pauli-overlap
  soft rules (`scl/symbolic.py`).
- **Process-engineering layer** (`scl/process.py`): synthesis-window survival
  + phase nucleation drift, plumbed through `scl/lab.py`. The loop now learns
  on the **realized** phase, not the requested one.

### Open threads (post-Milestone-2)
- NNQS proxy currently uses a heuristic (J, h) ↔ candidate mapping. Replace
  with a learned mapping calibrated against a small DFT/exact-diag dataset.
- Inverse design uses numerical gradients through the GP — analytic kernel
  derivatives would be ~10× faster.
- Process layer drift is symmetric; should be biased toward common stable
  polymorphs once we have a database of them.
- Persistence: still no JSONL log of histories. Needed before cross-run
  benchmarking is meaningful.
- World model is single-peak — add a multi-modal landscape to stress-test
  exploration vs the manifold/inverse-design machinery.

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
