"""LLM-driven hypothesizer agent.

Wraps Claude (via the Anthropic SDK) with a small toolbox built from the
existing scl modules. The agent reasons about what to try next instead of
relying on the hard-coded UCB / falsify / inverse-design cadences. Each round
the agent calls tools as needed and ends by invoking ``submit_to_lab`` with
its chosen candidate; the outer loop then runs the lab and resumes.

Defaults to ``claude-opus-4-7`` with adaptive thinking + ``xhigh`` effort —
the recommended setting for agentic / coding workloads. The system prompt and
tool schemas are stable across rounds, so prompt caching kicks in
automatically after the first call.

Exposed via the ``[agent]`` optional dependency group; this module imports
``anthropic`` lazily so the rest of the package keeps working without it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from .candidates import Candidate, ELEMENTS, METALS, featurize, sample_random
from .diffphys import inverse_design
from .falsify import falsify_neighbors
from .manifold import curvature
from .neural import GPSurrogate
from .nnqs import hubbard_proxy, quantum_proxy
from .symbolic import symbolic_check


_DEFAULT_MODEL = "claude-opus-4-7"


SYSTEM_PROMPT = """You are an autonomous scientific discovery agent searching
for room-temperature superconductor candidates in a hydride composition space
(metal + hydrogen at high pressure). Your goal is to maximize Tc (in Kelvin)
within a finite experiment budget. Each round you deliberate using the
available tools and must submit exactly ONE candidate to the lab via
submit_to_lab — that ends your turn.

The materials space:
- Composition: 2-3 components, ALWAYS including hydrogen (H) at fraction
  in (0.05, 0.99). Other elements: Li, B, C, Mg, S, Ca, Y, La, Ce.
- Pressure: 5-595 GPa. Higher pressures help superconductivity but
  synthesis becomes unreliable above ~300 GPa.
- Fractions must sum to 1.0.

The tools you have:
- web_search(query): Anthropic-hosted web search. USE THIS at cold start
  to ground your first few proposals in actual superconductor literature
  (recent ambient-pressure RTSC claims, hydride compositions reported at
  lower pressures, ternary additions that improved Tc). Don't web_search
  every round — once you have measurement data, the surrogate is the
  primary signal. One or two literature searches per run is plenty.
- web_fetch(url): Anthropic-hosted URL fetcher. Use to read the abstract
  or methods section of a paper web_search surfaced. Output is the raw
  page; pull out the composition / pressure / Tc and reason about whether
  it's worth trying a variant.
- propose_random_pool(n): Sample n random valid candidates. Useful for
  cold start or exploration when literature search doesn't surface a
  clear lead.
- symbolic_check(composition, pressure_gpa): First-principles validity
  check. Hard rules veto the candidate; soft rules are warnings. ALWAYS run
  before submit_to_lab.
- predict_tc(composition, pressure_gpa): Surrogate-model prediction
  (mean ± std). Returns warmup=true if the model has not been trained yet.
- manifold_curvature(composition, pressure_gpa): Curvature-of-belief at this
  candidate. Positive = peak-like (often promising).
- inverse_design(target_tc_k): Gradient-descent in feature space toward a
  target Tc; returns a candidate the surrogate predicts will hit it.
- falsify_probe(): Probe the model's most-confident-failure neighbor of the
  current best. If reality rewards it, the model is wrong somewhere
  important.
- quantum_proxy(composition, pressure_gpa): Per-site ground-state energy
  from a small NNQS (RBM over TFIM). Lower = stronger coupling regime.
  EXPENSIVE — only use on your top one or two picks.
- inspect_history(last_n): Get the last n measured rounds.
- submit_to_lab(composition, pressure_gpa): Submit the chosen candidate to
  the lab. ENDS YOUR TURN. Do not call any other tool in the same turn
  after this.

Strategy guidance:
- Cold start (no measurements yet): web_search the recent superconductor
  literature for compositions that recently improved Tc (or ambient-
  pressure claims), let that anchor your first 1-3 picks; fall back to
  propose_random_pool + symbolic_check if nothing useful surfaces.
- With data: balance EXPLOIT (highest predicted Tc), EXPLORE (highest
  uncertainty), and FALSIFY (probe what your model says will fail). Use
  inverse_design when you have a clear target and want to go directly
  toward it. Use quantum_proxy as a second opinion on your top pick.
- Always run symbolic_check on your final candidate before submit_to_lab.

Be terse. State your hypothesis for this round in one or two sentences,
then act. Do not narrate every tool call."""


TOOLS: list[dict[str, Any]] = [
    # Anthropic server-side tools — the agent uses these to ground hypotheses
    # in actual superconductor literature instead of relying on its training
    # data alone. Executed by Anthropic; we never see them in our tool runner.
    {"type": "web_search_20260209", "name": "web_search"},
    {"type": "web_fetch_20260209", "name": "web_fetch"},
    {
        "name": "propose_random_pool",
        "description": "Sample n random candidates that pass the hard symbolic checks. Returns a list of {composition, pressure_gpa, formula} objects.",
        "input_schema": {
            "type": "object",
            "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 30}},
            "required": ["n"],
        },
    },
    {
        "name": "symbolic_check",
        "description": "Run the System-2 verifier on a candidate. Returns ok=true/false plus a list of failed rules with severity hard/soft.",
        "input_schema": {
            "type": "object",
            "properties": {
                "composition": {
                    "type": "array",
                    "description": "List of [element_symbol, fraction] pairs.",
                    "items": {"type": "array"},
                },
                "pressure_gpa": {"type": "number"},
            },
            "required": ["composition", "pressure_gpa"],
        },
    },
    {
        "name": "predict_tc",
        "description": "Surrogate prediction (mean and std) for a candidate. Returns {warmup: true} if the model has not been trained yet.",
        "input_schema": {
            "type": "object",
            "properties": {
                "composition": {"type": "array", "items": {"type": "array"}},
                "pressure_gpa": {"type": "number"},
            },
            "required": ["composition", "pressure_gpa"],
        },
    },
    {
        "name": "manifold_curvature",
        "description": "Curvature-of-belief score for a candidate. Positive = peak-like (often a promising place to dig).",
        "input_schema": {
            "type": "object",
            "properties": {
                "composition": {"type": "array", "items": {"type": "array"}},
                "pressure_gpa": {"type": "number"},
            },
            "required": ["composition", "pressure_gpa"],
        },
    },
    {
        "name": "inverse_design",
        "description": "Gradient-descend the surrogate toward the target Tc, then project to a discrete candidate. Returns a candidate or null.",
        "input_schema": {
            "type": "object",
            "properties": {"target_tc_k": {"type": "number"}},
            "required": ["target_tc_k"],
        },
    },
    {
        "name": "falsify_probe",
        "description": "Generate an adversarial perturbation of the current best the surrogate predicts will FAIL. Returns a candidate or null if no measurements yet.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "quantum_proxy",
        "description": "NNQS variational ground-state energy per site (TFIM model). Approximate; serves as a coupling-regime indicator. Slow (~1s). Lower = stronger coupling regime. Use sparingly on top picks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "composition": {"type": "array", "items": {"type": "array"}},
                "pressure_gpa": {"type": "number"},
            },
            "required": ["composition", "pressure_gpa"],
        },
    },
    {
        "name": "hubbard_proxy",
        "description": "Exact-diagonalised Hubbard-model ground-state energy per site for a 4-site chain at half-filling. The candidate's features map to (t, U); the solver is exact (no variational error). Returns per-site energy: more negative = kinetic-energy regime (metallic, more conducive to phonon-mediated SC), positive = Mott-insulating regime. Faster and more rigorous than quantum_proxy for direct calibration.",
        "input_schema": {
            "type": "object",
            "properties": {
                "composition": {"type": "array", "items": {"type": "array"}},
                "pressure_gpa": {"type": "number"},
            },
            "required": ["composition", "pressure_gpa"],
        },
    },
    {
        "name": "inspect_history",
        "description": "Get the last n measured rounds with predicted/measured Tc.",
        "input_schema": {
            "type": "object",
            "properties": {"last_n": {"type": "integer", "minimum": 1, "maximum": 50}},
        },
    },
    {
        "name": "submit_to_lab",
        "description": "Submit the chosen candidate to the lab. ENDS YOUR TURN. Always run symbolic_check first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "composition": {"type": "array", "items": {"type": "array"}},
                "pressure_gpa": {"type": "number", "minimum": 0, "maximum": 600},
            },
            "required": ["composition", "pressure_gpa"],
        },
    },
]


def _serialize_candidate(c: Candidate) -> dict[str, Any]:
    return {
        "composition": [[e, float(f)] for e, f in c.composition],
        "pressure_gpa": float(c.pressure_gpa),
        "formula": c.formula(),
    }


def _parse_candidate(payload: dict[str, Any]) -> Candidate:
    """Best-effort parse of a tool input into a Candidate.

    Tolerates fractions that don't sum to exactly 1 (renormalises) and clips
    pressure to a sane range. Raises ValueError on truly broken inputs so the
    agent gets actionable feedback.
    """
    raw = payload.get("composition")
    if not isinstance(raw, list) or not raw:
        raise ValueError("composition must be a non-empty list of [element, fraction] pairs")

    entries: list[tuple[str, float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"composition entry {item!r} must be [element, fraction]")
        elem, frac = item
        elem = str(elem)
        if elem not in ELEMENTS:
            raise ValueError(f"unknown element {elem!r}; valid: {sorted(ELEMENTS)}")
        try:
            frac = float(frac)
        except (TypeError, ValueError) as e:
            raise ValueError(f"fraction for {elem} is not numeric: {e}") from None
        if not (0.0 < frac <= 1.0):
            raise ValueError(f"fraction for {elem} = {frac} outside (0, 1]")
        entries.append((elem, frac))

    total = sum(f for _, f in entries)
    if abs(total - 1.0) > 1e-3:
        entries = [(e, f / total) for e, f in entries]

    pressure = payload.get("pressure_gpa")
    try:
        pressure = float(pressure)
    except (TypeError, ValueError):
        raise ValueError(f"pressure_gpa must be numeric, got {pressure!r}") from None
    pressure = float(np.clip(pressure, 5.0, 595.0))

    return Candidate(composition=tuple(entries), pressure_gpa=pressure)


@dataclass
class AgentTools:
    """State the per-round tools close over."""

    model: GPSurrogate
    seen: list[Candidate]
    y_train: list[float]
    rng: np.random.Generator
    target_tc_k: float

    def execute(self, name: str, inp: dict[str, Any]) -> Any:
        method = getattr(self, f"_tool_{name}", None)
        if method is None:
            raise ValueError(f"unknown tool: {name}")
        return method(inp)

    def _tool_propose_random_pool(self, inp: dict[str, Any]) -> dict[str, Any]:
        n = int(inp.get("n", 5))
        n = max(1, min(n, 30))
        out: list[dict[str, Any]] = []
        attempts = 0
        while len(out) < n and attempts < n * 20:
            c = sample_random(self.rng)
            if symbolic_check(c).ok:
                out.append(_serialize_candidate(c))
            attempts += 1
        return {"candidates": out, "count": len(out)}

    def _tool_symbolic_check(self, inp: dict[str, Any]) -> dict[str, Any]:
        c = _parse_candidate(inp)
        res = symbolic_check(c)
        return {
            "ok": res.ok,
            "failures": [
                {"name": n, "severity": s, "message": m} for n, s, m in res.failures
            ],
        }

    def _tool_predict_tc(self, inp: dict[str, Any]) -> dict[str, Any]:
        c = _parse_candidate(inp)
        if self.model.X_train is None:
            return {"warmup": True, "message": "surrogate not trained yet"}
        feats = featurize(c)
        mu, sd = self.model.predict(feats)
        return {"mean_tc_k": float(mu[0]), "std_tc_k": float(sd[0])}

    def _tool_manifold_curvature(self, inp: dict[str, Any]) -> dict[str, Any]:
        c = _parse_candidate(inp)
        return {"curvature": float(curvature(c, self.model))}

    def _tool_inverse_design(self, inp: dict[str, Any]) -> dict[str, Any]:
        target = float(inp.get("target_tc_k", self.target_tc_k))
        c = inverse_design(target, self.model, self.rng, n_starts=4, steps=40)
        if c is None:
            return {"candidate": None, "message": "no valid candidate found"}
        return {"candidate": _serialize_candidate(c)}

    def _tool_falsify_probe(self, inp: dict[str, Any]) -> dict[str, Any]:
        if not self.y_train:
            return {"candidate": None, "message": "no measurements yet"}
        i = int(np.argmax(self.y_train))
        probe = falsify_neighbors(self.seen[i], self.model, self.rng)
        if probe is None:
            return {"candidate": None, "message": "no valid probe found"}
        return {"candidate": _serialize_candidate(probe)}

    def _tool_quantum_proxy(self, inp: dict[str, Any]) -> dict[str, Any]:
        c = _parse_candidate(inp)
        e = quantum_proxy(c, n_sites=4, n_hidden=4, steps=40, lr=0.05)
        return {"energy_per_site": float(e)}

    def _tool_hubbard_proxy(self, inp: dict[str, Any]) -> dict[str, Any]:
        c = _parse_candidate(inp)
        e = hubbard_proxy(c, n_sites=4)
        return {"energy_per_site": float(e), "model": "1D Hubbard, 4 sites, half-filling"}

    def _tool_inspect_history(self, inp: dict[str, Any]) -> dict[str, Any]:
        last_n = int(inp.get("last_n", 10))
        last_n = max(1, min(last_n, 50))
        rows = [
            {"candidate": _serialize_candidate(c), "tc_k": float(tc)}
            for c, tc in zip(self.seen[-last_n:], self.y_train[-last_n:])
        ]
        return {
            "rows": rows,
            "best_tc_k": float(max(self.y_train)) if self.y_train else None,
            "total_measurements": len(self.y_train),
        }


class LLMHypothesizer:
    """Drives a manual tool-use loop until the agent calls ``submit_to_lab``."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        effort: str = "xhigh",
        max_turns: int = 15,
        max_tokens: int = 4096,
        client: Any = None,
    ):
        self.model = model
        self.effort = effort
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "agent extras not installed; run: pip install -e '.[agent]'"
            ) from e
        self._client = anthropic.Anthropic()
        return self._client

    def propose(
        self,
        tools: AgentTools,
        round_idx: int,
        total_rounds: int,
    ) -> Optional[Candidate]:
        client = self._ensure_client()
        user_msg = self._build_user_message(tools, round_idx, total_rounds)
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_msg}]

        for _ in range(self.max_turns):
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=TOOLS,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                messages=messages,
            )

            tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]

            if response.stop_reason == "end_turn" and not tool_uses:
                return None
            if not tool_uses:
                return None

            messages.append({"role": "assistant", "content": response.content})

            tool_results: list[dict[str, Any]] = []
            for tu in tool_uses:
                if tu.name == "submit_to_lab":
                    try:
                        return _parse_candidate(dict(tu.input))
                    except Exception as e:
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tu.id,
                                "content": f"Invalid candidate: {e}. Re-check composition and pressure_gpa, then try submit_to_lab again.",
                                "is_error": True,
                            }
                        )
                        continue

                try:
                    result = tools.execute(tu.name, dict(tu.input))
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": json.dumps(result),
                        }
                    )
                except Exception as e:
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": f"Tool error: {type(e).__name__}: {e}",
                            "is_error": True,
                        }
                    )

            messages.append({"role": "user", "content": tool_results})

        return None

    def _build_user_message(
        self,
        tools: AgentTools,
        round_idx: int,
        total_rounds: int,
    ) -> str:
        n = len(tools.y_train)
        best_block = "no measurements yet"
        if n:
            i = int(np.argmax(tools.y_train))
            best = tools.seen[i]
            best_block = (
                f"{tools.y_train[i]:.1f}K — {best.formula()} @ "
                f"{best.pressure_gpa:.0f}GPa"
            )
        last = "  (none)"
        if n:
            recent = list(zip(tools.seen[-5:], tools.y_train[-5:]))
            last_lines = [
                f"  - {c.formula()} @ {c.pressure_gpa:.0f}GPa → {tc:.1f}K"
                for c, tc in recent
            ]
            last = "\n".join(last_lines)
        warmup = "trained" if tools.model.X_train is not None else "not trained yet"
        return (
            f"Round {round_idx + 1}/{total_rounds}. "
            f"Target Tc: {tools.target_tc_k:.0f}K.\n"
            f"Surrogate: {warmup}.\n"
            f"Measurements completed: {n}.\n"
            f"Best so far: {best_block}.\n"
            f"Recent rounds:\n{last}\n\n"
            "Decide on the next candidate to submit to the lab. Reason briefly, "
            "use tools as needed, then call submit_to_lab."
        )
