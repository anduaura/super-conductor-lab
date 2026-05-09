"""super-conductor-lab: closed-loop neuro-symbolic superconductor discovery (sim).

Architecture (mapped to the manifesto):

    System 1  - scl.neural        Gaussian-process "hunch" over composition features.
    System 2  - scl.symbolic      Hard-rule veto enforcing first-principles compliance.
    World     - scl.world_model   Hidden ground-truth Tc landscape (stands in for DFT).
    Lab       - scl.lab           Mock self-driving lab: synthesis + noisy measurement.
    Active    - scl.active        UCB selection of next experiment.
    Falsify   - scl.falsify       Adversarial probing of current best hypothesis.
    Loop      - scl.loop          Orchestrator that closes the loop.
"""

from .candidates import Candidate, ELEMENTS, featurize, sample_random
from .symbolic import symbolic_check, SymbolicResult
from .neural import GPSurrogate
from .lab import Lab, MeasurementResult
from .active import ucb_select
from .falsify import falsify_neighbors
from .loop import run_loop, LoopResult

__all__ = [
    "Candidate",
    "ELEMENTS",
    "featurize",
    "sample_random",
    "symbolic_check",
    "SymbolicResult",
    "GPSurrogate",
    "Lab",
    "MeasurementResult",
    "ucb_select",
    "falsify_neighbors",
    "run_loop",
    "LoopResult",
]
