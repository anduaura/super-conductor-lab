# The S-AGI Manifesto

The vision document this codebase implements against. The prose below is the
original framing for what a *Scientific AGI* (S-AGI) for cracking
room-temperature superconductivity would actually require. The "How this maps
to code" notes after each section pin every claim to a concrete module so the
manifesto stays a *spec* rather than a wishlist.

## North Star

**The ultimate goal of this project is the discovery of a room-temperature
superconductor at ambient pressure** — a material with Tc ≥ 293 K (20 °C)
that does not require diamond-anvil-cell pressures. We tackle as much as
possible in software first: replacing each toy stand-in in this repo
(`world_model.py`, `neural.py`, `nnqs.py`, `symbolic.py`) with a
physics-grounded equivalent before bridging to real autonomous synthesis.
Every milestone is measured against whether it moves us toward this target.
See the README's "Goal" section for the software-first replacement table and
definition of success.

## Why software-first

Three things software actually changes in the discovery pipeline:

1. **Inverse design.** Forward search ("if I mix X and Y, what happens?")
   explores a vanishingly small fraction of compositional space. Inverse
   search ("what composition gives me Tc ≥ 293 K at ambient pressure?")
   starts from the goal and walks backward through the surrogate. GNoME
   (2023) demonstrated the principle at scale: 2.2M predicted-stable
   crystals from a generative model + DFT validation, far more candidate
   diversity than human-led search produced in the prior decade.
   `scl/diffphys.py` is the toy version in this repo.

2. **Filtering at scale.** A trained crystal-graph GNN ranks ~10⁶
   candidates per minute. The point isn't "find the answer outright" —
   it's "narrow the answer space by 4–5 orders of magnitude before any
   physical experiment runs." The closed loop in `scl/loop.py` is this
   filter wired to a feedback mechanism; the bench in `docs/bench/` shows
   it does real work over random sampling even on a toy landscape.

3. **Stability + formal verification.** DFT formation energies + phonon
   spectra + symbolic checks (pymatgen-style charge balance, Goldschmidt
   tolerance, Pauli-overlap proxies) catch a substantial fraction of
   "looks promising but won't synthesize" candidates before lab time gets
   spent on them. `scl/symbolic.py` is the toy version; queued milestones
   replace it with real chemistry tooling.

## What software cannot do — the sim-to-real wall

Three categories where simulation systematically misleads, and which the
toy stand-ins in this repo do **not** model:

- **Impurities.** Real materials carry defect concentrations of
  10¹⁵–10¹⁹ /cm³. Single-atom defects can kill the superconducting gap.
  DFT can model dilute defects; real materials are not in the dilute limit.
- **Kinetic accessibility.** A material can be thermodynamically stable
  yet practically unmakeable — the synthesis pathway requires energies,
  timescales, or precursor purities no real reactor produces. Predicting
  *whether a recipe can be cooked* is much harder than predicting *whether
  the final product would survive once made*. This is where most "DFT
  predicts it works" → "lab can't make it" failures occur.
- **Mesoscale defects.** Grain boundaries, twin domains, dislocations
  emerge at length scales (nm–µm) above DFT and below continuum
  elasticity. Currently a known unknown across most of materials science.

## The honest size of the software lever

A defensible estimate of where software-first can take us before the lab
must take over:

- **Compositional space:** shrink ~10²³ atomic combinations to a shortlist
  of 10²–10³ candidates worth synthesizing — an 18–21 orders-of-magnitude
  reduction. This is the dominant lever.
- **Stability prediction:** correctly flag ~70% of "would not synthesize"
  candidates via formation energy + phonon spectra; the kinetic-
  accessibility failures slip through.
- **Property prediction:** correctly rank-order candidates by predicted Tc
  with strong correlation to truth, but with residuals large enough that
  the top-ranked candidate is rarely *the* best.

The right framing is **not** "software solves it, lab confirms it." It's:

> **Software determines what to test; the lab determines what's actually
> true.**

Both halves are load-bearing. The lab's role doesn't shrink as software
improves — it moves up the stack, from "screen 10⁶ random candidates" to
"characterize, replicate, and scale-up the 10² shortlist software handed
us." The reality gap is the true frontier. Software is the lever that
makes the gap small enough to cross.

---

## 0. Premise

Building an AGI for the express purpose of cracking room-temperature
superconductivity (or any "Grand Challenge") requires moving beyond the
"chatty autocomplete" phase of AI and into a closed-loop discovery engine.
Think of it as building a "System 2" scientist — one that doesn't just guess
the next word but reasons through the next experiment.

---

## Part I — The First Closed Loop

### 1. The Core Architecture: Neuro-Symbolic Integration

Current AI is great at "vibes" (neural intuition) but bad at "rules" (symbolic
logic). For physics, you need both.

- **Neural Layer (System 1).** Uses Graph Neural Networks (like GNoME) and
  Transformers to ingest the entirety of human scientific literature, crystal
  structures, and raw sensor data. Provides the "hunch" about which atomic
  structures might work.
- **Symbolic Layer (System 2).** A formal reasoning engine that understands
  the hard laws of thermodynamics and electromagnetism. If the Neural Layer
  suggests a material that violates the conservation of energy, the Symbolic
  Layer "vetoes" it immediately.

> **How this maps to code.** `scl/neural.py` is the System-1 surrogate (a
> Gaussian process — stand-in for a GNN/transformer family). `scl/symbolic.py`
> is the System-2 veto: a rule registry where every candidate must pass
> hydrogen-presence, charge-balance, pressure-bounds, formation-driving-force,
> and Pauli-overlap checks before the loop will even consider it.

### 2. The "World Model"

To solve the "glue" problem, the AGI needs a model of the world that isn't
just pixels or text — it needs a Physics-Informed World Model.

- **Grounded Training.** Instead of just training on the internet, train the
  AGI on the results of millions of Density Functional Theory (DFT)
  simulations and quantum Monte Carlo runs.
- **Information-Manifold Mapping.** The AGI is programmed to treat quantum
  states as geometric manifolds. It learns to "see" the curvature of electron
  correlations. It doesn't just look for a material; it looks for the
  *mathematical symmetry* that allows Cooper pairs to survive at 300 K.

> **How this maps to code.** `scl/world_model.py` is the hidden ground-truth
> Tc landscape — a hand-crafted DFT proxy that the surrogate can only access
> through the lab. `scl/manifold.py` computes the numerical Hessian of the
> surrogate's mean prediction to score curvature in feature space; the loop
> uses this as an acquisition bonus on UCB top picks.

### 3. The "Closed-Loop" Laboratory

An S-AGI is useless if it's trapped in a digital box. It must have *agency*
in the physical world.

- **Autonomous Hypothesis Generation.** The AGI writes its own code to run
  complex simulations.
- **Self-Driving Labs.** The AGI is connected via API to robotic synthesis
  stations. It suggests a new hydride composition, the robots mix and bake it,
  and the sensor data (resistivity, magnetic susceptibility) is fed back into
  the AGI's training loop in real-time.
- **Falsification Loops.** Unlike a human, the AGI doesn't get "attached" to
  a theory. It actively tries to *prove its own model wrong* to find the most
  robust version.

> **How this maps to code.** `scl/lab.py` is the mock self-driving lab:
> synthesis-window survival, phase nucleation drift, Gaussian measurement
> noise. `scl/active.py` runs UCB selection over the symbolic survivors.
> `scl/falsify.py` generates adversarial neighbors of the current best and
> submits the one the surrogate is *most confident will fail* — the cleanest
> falsification primitive that fits in 40 LOC.

### 4. Hardware Requirements

Simulating 10²³ electrons is the hard problem. To build this AGI you'd
likely need:

- **1-Bit Quantization.** 1-bit LLM architectures allow massive scaling with
  100× less energy, making it possible to run reasoning loops locally at
  the lab.
- **Quantum Kernels.** Use early-stage quantum processors as accelerators
  for many-body equations classical silicon can't handle.

> **How this maps to code.** Out of scope for this prototype — the simulator
> is numpy-only and runs in milliseconds per round. The architecture is
> deliberately structured so a real surrogate (`scl/neural.py`) or a real
> quantum-state engine (`scl/nnqs.py`) can be swapped in without touching the
> loop.

---

## Part II — The Virtual Brain

> Focusing on the *Virtual* layer is the strategic choice. The "brain" is
> currently the bottleneck, not the "hands". While we can build robots to
> mix chemicals all day, without a high-fidelity virtual engine we are just
> "failing faster" rather than "learning smarter".

### 5. Neural-Network Quantum States (NNQS)

The traditional way to simulate a material's behavior is to solve the
Schrödinger equation for 10²³ particles — a mathematical impossibility for
classical computers. Instead of brute-forcing the math, use the AGI to
represent the wavefunction Ψ as a deep neural network.

- **The logic.** Treat the quantum state as a high-dimensional optimization
  problem. By minimizing the energy functional `E[Ψ]`, the AGI "learns" the
  ground state of a potential superconductor without solving every
  interaction manually.
- **The math.** The AGI iteratively adjusts the weights of the network to
  find the global minimum, effectively *hallucinating the correct physics*
  before it ever enters a lab.

> **How this maps to code.** `scl/nnqs.py` implements a Carleo–Troyer RBM
> wavefunction `log Ψ(s) = a·s + Σ_h log cosh(b_h + W_h·s)` over the full
> Hilbert space of a small transverse-field Ising chain (N≤12 so the basis
> can be enumerated exactly). Analytic VMC gradients give the variational
> ground state in ~100 iterations. `quantum_proxy(c)` maps a candidate's
> features to (J, h) and returns per-site ground energy as a second-opinion
> score the loop calls every `nnqs_every` rounds.

### 6. The Information-Manifold Engine

This is where the AGI moves beyond simple simulation and into true discovery.
By mapping the electronic correlations of materials onto a
*Unitary Information-Manifold*, the AGI can identify geometric shortcuts to
superconductivity.

- **The concept.** Instead of looking at atoms, the AGI looks at the
  entanglement entropy between particles. It treats the transition to a
  superconducting state as a *topological collapse* of the manifold.
- **The benefit.** The AGI can predict *why* a material works. It can see
  the curvature of the electron interactions and suggest modifications to the
  crystal lattice that would straighten those curves, leading to higher-
  temperature stability.

> **How this maps to code.** `scl/manifold.py` computes a numerical-Hessian
> trace of the surrogate's predicted-Tc surface at each candidate point — a
> proxy for entanglement-entropy curvature. The loop's UCB selection is
> nudged by this manifold bonus, biasing exploration toward
> "topologically interesting" patches of the materials manifold.

### 7. Differentiable Physics & Formal Verification

A major risk with virtual discovery is the AI hallucinating a material that
works in its simulation but violates a hard law of physics in reality.

- **Differentiable physics.** Build the simulation such that the laws of
  physics are *baked in* as constraints. The AGI can backpropagate through
  the physical laws — it can ask "to get a Tc of 300 K, what must the atomic
  mass of the dopant be?" and get a physically valid answer.
- **Symbolic verification.** Every "hunch" the neural network has is run
  through a symbolic checker — a rigid math engine that ensures the proposed
  material doesn't violate things like the Pauli Exclusion Principle or basic
  thermodynamics.

> **How this maps to code.** `scl/diffphys.py` runs gradient descent in 7-D
> feature space toward a target Tc using numerical gradients through the
> surrogate, then projects the result back to a discrete (composition,
> pressure) candidate by closest-element-mix matching. Every output is run
> through `scl/symbolic.py` before it's allowed into the lab queue. The
> symbolic verifier carries both the hard physics rules and the
> Pauli-overlap / formation-driving-force soft rules.

### 8. The Outcome — A "Digital Twin" of Chemistry

By focusing on the Virtual, you create a system that can "test" a billion
materials in a single afternoon. When the AGI finally says "build this
specific copper-hydride at 1.2 atmospheres," you aren't just guessing —
you're executing a mathematically verified blueprint.

> **How this maps to code.** Today the prototype's "afternoon" is a few
> seconds: `scl run --rounds 200` exercises the full closed loop —
> System-1 hunch + System-2 veto + manifold bonus + falsification + inverse
> design + NNQS second opinion + process-aware lab — across hundreds of
> candidates, persists the trajectory, and reports a verified leader. The
> web UI (`scl serve`) renders the same trajectory live.

---

## Part III — Open Problems

Synthesis isn't simply "the easy part" once the brain is solved. Impurities
and heat loss aren't noise you can average out — they often pick *which phase
actually nucleates*. The brain's "build this" becomes "build this AND survive
a process window where 90% of attempts crystallize as a different polymorph".
The hard discipline shifts from theory to *process engineering* — what
synthesis path gets the structure into tolerance — and that's a separate AGI
problem the virtual brain doesn't automatically solve.

> **How this maps to code.** `scl/process.py` models the gap between
> *requested* and *realized* candidates: a synthesis-window survival
> probability and a phase-nucleation drift function. The lab returns the
> realized phase, not the requested one, so the loop must learn "makeable"
> alongside "good".

---

## Roadmap

The codebase has shipped through three milestones; four more are queued in
`CLAUDE.md`.

| Milestone | Status | Summary |
| --------- | ------ | ------- |
| 1. Closed-loop scaffold | ✅ done | GP + symbolic + UCB + falsification + mock lab |
| 2. Virtual-brain pillars | ✅ done | NNQS, manifold, diffphys, process layer |
| 3. Web UI | ✅ done | FastAPI + SSE + JSONL + Codespaces live demo |
| 4. LLM-driven hypothesizer agent | ⏳ next | Anthropic-SDK proposer that uses the existing modules as tools |
| 5. Multi-modal landscape + benchmarking harness | ⏳ queued | Make discovery genuinely hard; head-to-head acquisition comparison |
| 6. Run comparison + export in UI | ⏳ queued | Pick any two past runs, overlay, diff, export CSV/JSON |
| 7. Production hardening + Fly.io deploy | ⏳ queued | Token auth, Dockerfile, GitHub Actions, permanent shared URL |

---

## Building philosophy

Three rules that keep this prototype honest:

1. **The world model is hidden.** Only `scl/lab.py` and tests may import
   `world_model.true_tc`. Every other module reasons through the surrogate.
   This is what makes "the loop is actually learning" a falsifiable claim
   rather than a vibe.
2. **Symbolic veto runs before everything.** A candidate that violates a hard
   rule never reaches the lab, the surrogate, the manifold scorer, or the
   inverse designer. This is what keeps the loop from spending experiment
   budget on garbage.
3. **Numpy-only in the core.** Web tooling (`fastapi`, `uvicorn`) lives behind
   the `[web]` extra. A real ML stack (torch / scipy / sklearn) is *not*
   required to demonstrate the architecture; adding it later is a swap, not
   a rewrite.
