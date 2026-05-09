"""Composition candidates and featurization.

The chemical space is a small slice of the superhydride landscape: a metal (or
two) plus hydrogen, under high pressure. Featurization collapses each candidate
to a fixed-length vector consumed by both the surrogate model and the symbolic
rule engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Subset of elements relevant to superhydride exploration.
# valence is the nominal oxidation state used for the charge-balance heuristic.
ELEMENTS: dict[str, dict[str, float]] = {
    "H":  {"Z":  1, "mass":   1.008, "EN": 2.20, "radius":  25, "valence":  1},
    "Li": {"Z":  3, "mass":   6.941, "EN": 0.98, "radius": 152, "valence":  1},
    "B":  {"Z":  5, "mass":  10.811, "EN": 2.04, "radius":  85, "valence":  3},
    "C":  {"Z":  6, "mass":  12.011, "EN": 2.55, "radius":  70, "valence":  4},
    "Mg": {"Z": 12, "mass":  24.305, "EN": 1.31, "radius": 150, "valence":  2},
    "S":  {"Z": 16, "mass":  32.060, "EN": 2.58, "radius": 100, "valence": -2},
    "Ca": {"Z": 20, "mass":  40.078, "EN": 1.00, "radius": 197, "valence":  2},
    "Y":  {"Z": 39, "mass":  88.906, "EN": 1.22, "radius": 180, "valence":  3},
    "La": {"Z": 57, "mass": 138.905, "EN": 1.10, "radius": 187, "valence":  3},
    "Ce": {"Z": 58, "mass": 140.116, "EN": 1.12, "radius": 182, "valence":  3},
}

METALS: tuple[str, ...] = tuple(e for e in ELEMENTS if e != "H")

FEATURE_NAMES: tuple[str, ...] = (
    "avg_mass", "avg_en", "avg_radius", "avg_valence",
    "h_fraction", "en_diff", "pressure_gpa",
)


@dataclass(frozen=True)
class Candidate:
    """A composition + synthesis pressure."""

    composition: Tuple[Tuple[str, float], ...]
    pressure_gpa: float

    def formula(self) -> str:
        return " ".join(f"{e}{f:.2f}" for e, f in self.composition)

    def h_fraction(self) -> float:
        return next((f for e, f in self.composition if e == "H"), 0.0)


def featurize(c: Candidate) -> np.ndarray:
    avg_mass = sum(f * ELEMENTS[e]["mass"] for e, f in c.composition)
    avg_en = sum(f * ELEMENTS[e]["EN"] for e, f in c.composition)
    avg_radius = sum(f * ELEMENTS[e]["radius"] for e, f in c.composition)
    avg_val = sum(f * ELEMENTS[e]["valence"] for e, f in c.composition)
    h_frac = c.h_fraction()
    ens = [ELEMENTS[e]["EN"] for e, _ in c.composition]
    en_diff = max(ens) - min(ens) if len(ens) > 1 else 0.0
    return np.array(
        [avg_mass, avg_en, avg_radius, avg_val, h_frac, en_diff, c.pressure_gpa],
        dtype=float,
    )


def sample_random(rng: np.random.Generator) -> Candidate:
    """Sample a random binary or ternary metal-hydride at a random pressure."""
    n_components = int(rng.choice([2, 3]))
    h_frac = float(rng.uniform(0.10, 0.95))
    if n_components == 2:
        m = str(rng.choice(METALS))
        comp = ((m, 1.0 - h_frac), ("H", h_frac))
    else:
        idx = rng.choice(len(METALS), size=2, replace=False)
        m1, m2 = METALS[int(idx[0])], METALS[int(idx[1])]
        # split remaining (1 - h_frac) between two metals, never letting either go below 5%.
        rem = 1.0 - h_frac
        f_m1 = float(rng.uniform(0.05 * rem, 0.95 * rem))
        f_m2 = rem - f_m1
        comp = ((m1, f_m1), (m2, f_m2), ("H", h_frac))
    pressure = float(rng.uniform(20.0, 500.0))
    return Candidate(composition=comp, pressure_gpa=pressure)


def perturb(c: Candidate, rng: np.random.Generator, scale: float = 0.05) -> Candidate:
    """Small composition + pressure perturbation, renormalized to a valid candidate."""
    fracs = np.array([f for _, f in c.composition], dtype=float)
    elems = [e for e, _ in c.composition]
    noise = rng.normal(0.0, scale, size=fracs.shape)
    new = np.clip(fracs + noise, 0.01, 1.0)
    new = new / new.sum()
    new_comp = tuple((e, float(f)) for e, f in zip(elems, new))
    new_p = float(np.clip(c.pressure_gpa + rng.normal(0.0, 25.0), 5.0, 600.0))
    return Candidate(composition=new_comp, pressure_gpa=new_p)
