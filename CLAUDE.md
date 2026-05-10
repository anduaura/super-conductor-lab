# CLAUDE.md

Operating notes for Claude when working in this repo.

## Project

`super-conductor-lab` is a numpy-only simulator of a closed-loop neuro-symbolic
discovery engine for room-temperature superconductor candidates. It maps the
S-AGI manifesto to runnable code without depending on real DFT, GPUs, or a real
self-driving lab.

## North star

The ultimate goal is **the discovery of a room-temperature superconductor at
ambient pressure** (Tc ≥ 293 K, no diamond-anvil cell required). Every
milestone should be evaluated against whether it moves us toward that —
typically by replacing a toy stand-in (Gaussian world model, GP surrogate,
TFIM proxy, hand-coded rules) with a physics-grounded equivalent, sharpening
the candidate-ranking pipeline, or building the bridge to a real autonomous
synthesis lab. See the README "Goal" section for the software-first
replacement plan.

**Known conceptual tension to fix.** The current `world_model.py` rewards
high pressure as a Tc multiplier, which models the high-pressure-hydride
regime (LaH₁₀ etc.). The actual goal is *ambient-pressure* RTSC, so a
follow-up should split the pressure axis into "synthesis pressure" (helps
form the material) vs "operating pressure" (must be ~1 atm for the goal).

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

### Milestone 4 — LLM hypothesizer agent (done)
- `scl/agent.py`: `LLMHypothesizer` drives a manual tool-use loop against the
  Anthropic SDK. `AgentTools` exposes the existing modules as 9 tools
  (`propose_random_pool`, `symbolic_check`, `predict_tc`, `manifold_curvature`,
  `inverse_design`, `falsify_probe`, `quantum_proxy`, `inspect_history`,
  `submit_to_lab`). The agent picks one candidate per round; `submit_to_lab`
  ends its turn and the loop runs the lab.
- Defaults: `claude-opus-4-7`, `thinking={type: "adaptive"}`,
  `effort="xhigh"`. System prompt + tool schemas are stable and cached via
  `cache_control: ephemeral`. Falls back to UCB on any agent error.
- Wired into `scl/loop.py` (`use_agent=True` subsumes the falsify/inverse/
  nnqs cadences), `scl/cli.py` (`--use-agent`, `--agent-model`,
  `--agent-effort`), and the web UI form.
- Behind the `[agent]` optional dependency group (`pip install -e '.[agent]'`).
  Tests mock the SDK client — no real API calls in CI.

### Milestone 3 — web UI (done)
- FastAPI backend (`scl/web/app.py`) with REST + SSE endpoints.
- `RunManager` (`scl/web/runner.py`) executes each run in a background thread,
  fans out events via a snapshot-cursor SSE pattern, persists completed runs
  via `RunStore` (`scl/web/storage.py`) as one JSONL file per run.
- Single-page vanilla-JS frontend (`scl/web/static/`): config form, live
  best-so-far line chart, predicted-vs-measured scatter, round log, history
  table with replay-on-click, optional paired-baseline overlay.
- `scl serve --port 8765` starts uvicorn on the local host.
- Web layer is behind the `[web]` optional dep group; importable but not
  required for the core CLI.

### Milestone 5 — multi-modal landscape + benchmark harness (done)
- `scl/world_model.py` — added `mode="multi"` parameter; multi-mode is the
  sum of four Gaussian peaks at distinct `(h_frac, pressure, en_diff,
  avg_val)` centers (220 K / 270 K / 260 K / 320 K), the highest one in a
  narrow attractor at an unusual valence. `mode="single"` still returns the
  original unimodal landscape.
- `scl/lab.py` and `scl/loop.py` — accept a `world_mode` parameter and
  thread it through `true_tc`. Default stays `"single"` for backward
  compatibility.
- `scl/active.py` — added `ei_select` (Expected Improvement) and
  `thompson_select` (marginal Thompson sampling) alongside the existing
  UCB. `run_loop` now takes `acquisition="ucb"|"ei"|"thompson"`.
- `scl/bench.py` — strategy × seed grid harness. Eight named strategies
  (random, ucb, ei, thompson, ucb+manifold, ucb+falsify, ucb+inverse, all).
  CSV writer + median/IQR summary table.
- `scl bench` CLI subcommand drives the grid; results in `docs/bench/`
  (single.csv, multi.csv, README.md analysis).

Headline findings on the multi-modal landscape: random search hits 152 K
median; UCB+manifold leads at 205 K. No strategy reliably finds Peak D
(320 K) in 30 rounds — by design, this is the open problem the closed loop
motivates rather than solves.

### Milestone 6 — run comparison + export in web UI (done)
- Two new export endpoints: `GET /api/runs/{id}/export.json` and `.csv`,
  both with `Content-Disposition: attachment` so the browser downloads the
  file directly.
- History table gained a checkbox column and a "compare selected" button —
  pick any number of past runs and overlay their `best so far` and
  `predicted vs measured` series in a rotating color palette. Per-row
  `CSV` / `JSON` links sit alongside the existing `view` button.
- `static/app.js` was rewritten to manage chart datasets dynamically
  (`ensureSeries(label, color)`) instead of two hard-coded series, so any
  number of runs can be overlaid simultaneously.
- 3 new pytests cover the export endpoints (JSON round-trip, CSV columns,
  404 on unknown id). 61 pytests passing.

### Milestone 7 — production hardening (done)
- **Auth.** When `SCL_AUTH_TOKEN` env var is set (or `--auth-token` is
  passed to `scl serve`), every `/api/*` request requires
  `Authorization: Bearer <token>`, or `?token=<token>` query parameter for
  SSE (which can't send headers). `/healthz` and `/static/*` stay open so
  the UI can bootstrap and prompt for the token. Token comparison uses
  `hmac.compare_digest` to avoid timing leaks.
- **Frontend.** `static/app.js` reads/writes the token from
  `localStorage`, attaches it to every `fetch`, and inserts `?token=` on
  EventSource URLs and `<a>` export links. On a 401 the UI prompts for the
  token and reloads.
- **Container.** `Dockerfile` (Python 3.11 slim, ~150 MB) installs the
  package with the `[web]` extra, exposes 8765, runs `scl serve` with
  `--host 0.0.0.0`. `.dockerignore` excludes test/build artifacts.
- **Fly.io.** `fly.toml` mounts a `scl_data` volume at `/data` for run
  persistence, healthchecks `/healthz`, auto-stops when idle. README
  documents the `fly launch` → `fly secrets set SCL_AUTH_TOKEN` →
  `fly deploy` path.
- 6 new pytests cover the auth surface (open paths, blocked API,
  Bearer header, query-param fallback, static-files-open). 67 pytests
  passing total.

### Milestone 8 — split synthesis vs operating pressure (done)
- `scl/world_model.py` — added `mode="ambient"` that evaluates Tc at
  operating pressure ≈ 1 atm regardless of the candidate's synthesis
  pressure. Synthesis pressure (the `pressure_gpa` field) still drives
  `scl/process.synthesis_window` and phase nucleation drift in `lab.py`,
  but no longer enters the Tc formula. Four ambient peaks placed at
  (h_frac, en_diff, avg_val) only; highest is 305 K at unusual valence.
- Threaded `--world-mode ambient` through `scl/cli.py` and the bench
  harness.
- 4 new pytests covering the ambient mode (ignores pressure, kills the
  multi-mode winner, attainable Tc range). 71 pytests passing.

Headline ambient-mode bench (`docs/bench/ambient.csv`, 8 strategies × 10
seeds, 30 rounds): UCB hits 237.8 K median / 311.3 K best, UCB+manifold
317.9 K best, random 112.6 K median. Three strategies cleared the 293 K
RTSC threshold on individual seeds — but never consistently across all
seeds. Reliability is the open problem M9–M12 try to close.

### Milestone 9 — pymatgen-grounded symbolic verifier (queued)

### Milestone 9 — pymatgen-grounded symbolic verifier (queued)
Replace the 5 hand-coded soft rules in `scl/symbolic.py` with real
chemistry checks via pymatgen + matminer: charge balance, Goldschmidt
tolerance factor, formation-energy heuristic against a reference database,
Pauli/Madelung sanity. Keep the rule registry pattern; add `[chem]`
optional dependency group. ~150 LOC + adds two well-known deps (with
`pip install -e '.[chem]'` opt-in so the core stays numpy-only).

### Milestone 10 — literature-search tool for the LLM agent (queued)
Add `web_search` and `web_fetch` Anthropic server-side tools to
`scl/agent.py` so the hypothesizer can read the actual superconductor
literature when proposing candidates ("LaH10-style hydrides at lower
pressure", "look for ternary carbon hydrides recently reported"). Cheapest
real-world grounding step; doesn't require any new ML stack.

### Milestone 11 — crystal-graph GNN surrogate (queued)
Replace `scl/neural.py` (toy GP) with a small graph neural network over
crystal structures, trained on a public materials dataset (Materials
Project's superconductor subset, ~16K labeled compositions). Adds a real
ML stack (`torch`) behind a `[gnn]` optional dependency. Big lift, biggest
fidelity gain. The closed-loop machinery (UCB / EI / Thompson, manifold
curvature, NNQS proxy, falsification) is structurally compatible — it
takes mean+std from the surrogate, doesn't care if the surrogate is a GP
or a GNN with MC dropout.

### Milestone 12 — calibrate NNQS against real exact-diag (queued)
Today `scl/nnqs.py` solves a 6-site TFIM with a heuristic
`(J, h) ↔ candidate-features` mapping. Replace with a learned mapping
calibrated against exact-diagonalization of small Hubbard models (6–8
sites) with realistic hopping/Coulomb parameters drawn from DFT for the
candidate's elements. Turns the "quantum proxy" from a heuristic into a
real second opinion. Scientific-depth lift; opens the door to extending
to larger lattices via VMC sampling.

### Open threads (post-Milestone-7)
- Auth is single-token only — no per-user accounts, no rotation, no
  scopes. Fine for a personal deploy; would need real auth for sharing.
- No rate limiting on the API. A malicious authed user could spam
  `POST /api/runs` and burn machine time.
- Run history is unbounded — no GC. Persistent volumes will fill up
  eventually.

### Open threads (post-Milestone-6)
- Compare view doesn't show a delta/diff summary table — just overlaid
  charts. A small "leaderboard" panel listing each compared run's best Tc
  + composition would close that.
- Export is per-run only; no bulk export of "all runs in this week."
- Frontend chart dataset accumulation is unbounded — if you compare 50+
  runs the canvas legend gets unreadable. Add a hard cap.

### Open threads (post-Milestone-5)
- Multi-modal landscape is hand-tuned. A reproducibility study against a
  real DFT/exact-diag dataset would replace the heuristic peak placement.
- "all" strategy underperforms on smooth landscapes — needs an adaptive
  schedule (run more falsification when surrogate uncertainty is high,
  not on a fixed cadence).
- No EI-with-manifold or Thompson-with-manifold combinations yet — the
  `_ucb_with_manifold` helper is hard-coded to UCB.
- Bench results are CSV only — a static visualization page in
  `docs/bench/` that loads the CSVs and renders box plots would be a
  natural milestone-6 follow-on.

### Open threads (post-Milestone-3)
- No auth on the web UI — fine for localhost, not for sharing. Add a token
  if we ever expose it.
- SSE polls every 100ms; could be reduced via `threading.Event`-bridged
  asyncio if we ever care about latency.
- History view replays the full event list each time; for thousands of
  rounds this becomes O(N) JSON. Add a `?cursor=` parameter when needed.

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
