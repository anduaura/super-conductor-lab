"""super-conductor-lab: closed-loop neuro-symbolic superconductor discovery (sim).

Architecture (mapped to the manifesto):

    System 1   - scl.neural        Gaussian-process 'hunch' over composition features.
    System 2   - scl.symbolic      Hard-rule veto + Pauli/thermo soft rules.
    World      - scl.world_model   Hidden ground-truth Tc landscape (DFT stand-in).
    Process    - scl.process       Synthesis-window survival + phase nucleation drift.
    Lab        - scl.lab           Mock self-driving lab using the process layer.
    Active     - scl.active        UCB selection of next experiment.
    Manifold   - scl.manifold      Curvature-of-belief acquisition bonus.
    Falsify    - scl.falsify       Adversarial probing of current best hypothesis.
    NNQS       - scl.nnqs          RBM wavefunction (TFIM) — quantum-proxy second opinion.
    DiffPhys   - scl.diffphys      Inverse design via gradient descent on the surrogate.
    Loop       - scl.loop          Orchestrator that closes the loop.
"""

from .active import random_select, ucb_select
from .candidates import Candidate, ELEMENTS, featurize, perturb, sample_random
from .diffphys import inverse_design
from .falsify import falsify_neighbors
from .lab import Lab, MeasurementResult
from .loop import LoopResult, run_loop
from .manifold import curvature, manifold_bonus
from .neural import GPSurrogate
from .nnqs import RBMWavefunction, exact_ground_energy, quantum_proxy
from .process import realized_phase, synthesis_window
from .symbolic import SymbolicResult, symbolic_check

__all__ = [
    "Candidate", "ELEMENTS", "featurize", "perturb", "sample_random",
    "symbolic_check", "SymbolicResult",
    "GPSurrogate",
    "Lab", "MeasurementResult",
    "ucb_select", "random_select",
    "manifold_bonus", "curvature",
    "falsify_neighbors",
    "RBMWavefunction", "exact_ground_energy", "quantum_proxy",
    "inverse_design",
    "synthesis_window", "realized_phase",
    "run_loop", "LoopResult",
]
