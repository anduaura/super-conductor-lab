# super-conductor-lab

A small, runnable prototype of a closed-loop neuro-symbolic discovery engine
for room-temperature superconductor candidates. It is the architecture in the
S-AGI manifesto compressed into a single Python package you can actually run.

This is a *simulator*. There is no real DFT, no real lab, no real GNN —
everything is replaced by a numpy-only stand-in that exhibits the same
control-flow shape so the loop can be exercised end to end.

## Architecture mapping

| Manifesto component        | Module               | Stand-in                           |
| -------------------------- | -------------------- | ---------------------------------- |
| Neural intuition (System 1)| `scl.neural`         | Gaussian-process surrogate         |
| Symbolic veto (System 2)   | `scl.symbolic`       | Rule engine (hard + soft rules)    |
| Physics-informed world     | `scl.world_model`    | Hand-crafted Tc landscape          |
| Self-driving lab           | `scl.lab`            | Mock synthesis + noisy measurement |
| Active learning            | `scl.active`         | UCB acquisition                    |
| Falsification              | `scl.falsify`        | Adversarial probe of current best  |
| Closed loop                | `scl.loop`           | Orchestrator                       |

## Run

    pip install -e .
    scl run --rounds 30 --seed 42 --baseline

`--baseline` runs an equivalent random-search loop for comparison.

## Test

    pip install -e '.[dev]'
    pytest -q
