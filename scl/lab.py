"""Mock self-driving lab.

A real implementation would dispatch to a robotic synthesis station and parse
back resistivity / magnetic-susceptibility traces. Here we wrap the hidden
world model with synthesis failure and Gaussian measurement noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .candidates import Candidate, featurize
from .world_model import true_tc


@dataclass
class MeasurementResult:
    candidate: Candidate
    success: bool
    tc_k: Optional[float]
    note: str = ""


class Lab:
    """Stateful mock lab. Tracks every experiment requested."""

    def __init__(
        self,
        rng: np.random.Generator,
        noise_k: float = 5.0,
        base_failure_prob: float = 0.05,
        high_pressure_failure_prob: float = 0.40,
        pressure_failure_threshold_gpa: float = 350.0,
    ):
        self.rng = rng
        self.noise_k = noise_k
        self.base_failure_prob = base_failure_prob
        self.high_pressure_failure_prob = high_pressure_failure_prob
        self.pressure_failure_threshold_gpa = pressure_failure_threshold_gpa
        self.history: list[MeasurementResult] = []

    def run(self, c: Candidate) -> MeasurementResult:
        # Synthesis at extreme pressures is unreliable.
        feats = featurize(c)
        pressure = float(feats[6])
        fail_prob = self.base_failure_prob + (
            self.high_pressure_failure_prob
            if pressure > self.pressure_failure_threshold_gpa
            else 0.0
        )
        if self.rng.random() < fail_prob:
            res = MeasurementResult(candidate=c, success=False, tc_k=None,
                                    note="synthesis failed")
            self.history.append(res)
            return res

        true = true_tc(c)
        observed = max(0.0, true + float(self.rng.normal(0.0, self.noise_k)))
        res = MeasurementResult(candidate=c, success=True, tc_k=observed)
        self.history.append(res)
        return res
