# super-conductor-lab

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/anduaura/super-conductor-lab)

A runnable, numpy-only prototype of a closed-loop neuro-symbolic discovery
engine for room-temperature superconductor candidates. The whole architecture
fits in one Python package; nothing is mocked at the API surface — only the
underlying physics is.

![super-conductor-lab UI](docs/ui.svg)

## Live demo

Click the Codespaces badge above. The container will:

1. Install the package with the `[web,dev]` extras.
2. Run the test suite to verify the build (~10s).
3. Start `scl serve` on port 8765 in the background.
4. Auto-open the forwarded URL in your browser.

You can then start a run from the form, watch it stream round-by-round, and
optionally launch a paired random-search baseline for comparison.

### Keeping it free

Codespaces costs nothing for typical demo use, but the only **hard** guarantee
is a spending limit. Three steps:

1. **Set a $0 spending limit (one-time, hard cap).** GitHub → Settings →
   Billing & plans → [Codespaces spending limit](https://github.com/settings/billing/spending_limit)
   → set to **$0**. With this in place, GitHub will refuse to bill you no
   matter what — usage simply stops once the free tier is exhausted.
2. **The 2-core default is enough.** Personal accounts get 120 core-hours and
   15 GB-month of storage free, which is ~60 hours on a 2-core machine. A
   full demo run is minutes, not hours.
3. **Stop the codespace when done.** Top-right menu → *Stop codespace*. The
   default 30-minute idle timeout does this automatically, but stopping
   manually keeps the meter at zero.

The "Codespace usage for this repository is paid for by anduaura" notice on
the launch page just routes any billing to the repo owner's account — the
$0 limit on that account means nothing can actually be charged.

## Architecture

| Manifesto component                | Module               | Stand-in                                |
| ---------------------------------- | -------------------- | --------------------------------------- |
| System 1 — neural intuition        | `scl/neural.py`      | Gaussian-process surrogate (mean + var) |
| System 2 — symbolic veto           | `scl/symbolic.py`    | Hard + soft rule engine                 |
| Hidden ground truth (DFT proxy)    | `scl/world_model.py` | Hand-crafted Tc landscape               |
| NNQS — quantum proxy               | `scl/nnqs.py`        | Carleo–Troyer RBM over a small TFIM     |
| Information-manifold engine        | `scl/manifold.py`    | Numerical Hessian curvature bonus       |
| Differentiable physics             | `scl/diffphys.py`    | Inverse design via gradient descent     |
| Process layer (synth + drift)      | `scl/process.py`     | Synthesis-window + phase nucleation     |
| Mock self-driving lab              | `scl/lab.py`         | Process-aware noisy measurement         |
| UCB acquisition                    | `scl/active.py`      | Exploit-explore selection               |
| Falsification probes               | `scl/falsify.py`     | Adversarial neighbors of current best   |
| Closed-loop driver                 | `scl/loop.py`        | Orchestrator with pluggable cadences    |
| Web UI (FastAPI + SSE)             | `scl/web/`           | REST + SSE + JSONL persistence          |
| CLI                                | `scl/cli.py`         | `scl run` and `scl serve`               |

The world model is **hidden from the surrogate** — the lab is the only data
channel. See `CLAUDE.md` for the full set of architectural invariants.

## Install

```bash
# core only — CLI + tests
pip install -e '.[dev]'

# with the web UI
pip install -e '.[web,dev]'
```

## Run

```bash
# headless: 30 rounds of active learning, optionally paired with random search
scl run --rounds 30 --seed 42 --baseline

# full UI on http://127.0.0.1:8765
scl serve --port 8765
```

CLI flags map 1:1 to loop knobs: `--kappa`, `--manifold-weight`,
`--falsify-every`, `--inverse-every`, `--nnqs-every`, `--target-tc`.

## Test

```bash
pytest -q
# 31 passing across symbolic, neural, NNQS, manifold, diffphys, process,
# loop, and web layers.
```

## Repository layout

```
scl/
├── candidates.py        # composition space + featurization
├── symbolic.py          # rule engine (System 2)
├── neural.py            # GP surrogate (System 1)
├── world_model.py       # hidden ground truth (DFT proxy)
├── nnqs.py              # RBM wavefunction (TFIM) + quantum_proxy
├── manifold.py          # curvature-of-belief acquisition bonus
├── diffphys.py          # inverse design
├── process.py           # synthesis window + phase drift
├── lab.py               # mock self-driving lab
├── active.py            # UCB / random selection
├── falsify.py           # adversarial probes
├── loop.py              # closed-loop orchestrator
├── cli.py               # `scl run` + `scl serve`
└── web/                 # FastAPI app + SSE + vanilla-JS frontend
tests/                   # 31 pytests
docs/ui.svg              # UI mock used in this README
CLAUDE.md                # operating rules + milestones
```
