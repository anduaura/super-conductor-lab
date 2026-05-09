"""Mock self-driving lab.

A real implementation would dispatch to a robotic synthesis station and parse
back resistivity / magnetic-susceptibility traces. Here we wrap the hidden
world model with a process-engineering layer (synthesis-window survival +
phase nucleation drift) and Gaussian measurement noise.

The lab returns the *realized* candidate, which can differ from what was
requested — the loop must learn 'makeable' alongside 'good'.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .candidates import Candidate
from .process import realized_phase, synthesis_window
from .world_model import true_tc


@dataclass
class MeasurementResult:
    candidate: Candidate          # actually realized phase (may differ from requested)
    requested: Candidate          # what the loop asked for
    success: bool
    tc_k: Optional[float]
    note: str = ""


class Lab:
    """Stateful mock lab. Tracks every experiment requested."""

    def __init__(self, rng: np.random.Generator, noise_k: float = 5.0):
        self.rng = rng
        self.noise_k = noise_k
        self.history: list[MeasurementResult] = []

    def run(self, c: Candidate) -> MeasurementResult:
        if self.rng.random() > synthesis_window(c):
            res = MeasurementResult(
                candidate=c, requested=c, success=False, tc_k=None,
                note="outside synthesis window",
            )
            self.history.append(res)
            return res

        realized = realized_phase(c, self.rng)
        observed = max(
            0.0,
            true_tc(realized) + float(self.rng.normal(0.0, self.noise_k)),
        )
        note = "phase drift" if realized is not c else ""
        res = MeasurementResult(
            candidate=realized, requested=c, success=True,
            tc_k=observed, note=note,
        )
        self.history.append(res)
        return res
