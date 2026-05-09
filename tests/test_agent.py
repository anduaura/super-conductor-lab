"""Tests for scl.agent — uses a fake Anthropic client (no real API calls)."""

from types import SimpleNamespace

import numpy as np
import pytest

from scl.agent import (
    LLMHypothesizer,
    AgentTools,
    SYSTEM_PROMPT,
    TOOLS,
    _parse_candidate,
    _serialize_candidate,
)
from scl.candidates import Candidate, featurize, sample_random
from scl.neural import GPSurrogate
from scl.symbolic import symbolic_check


def _tool_use(name, input_data, tool_id):
    return SimpleNamespace(type="tool_use", name=name, input=input_data, id=tool_id)


def _response(blocks, stop_reason="tool_use"):
    return SimpleNamespace(content=blocks, stop_reason=stop_reason)


class FakeMessages:
    """Records the calls and returns scripted responses."""

    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._scripted:
            raise AssertionError("FakeMessages: ran out of scripted responses")
        return self._scripted.pop(0)


class FakeClient:
    def __init__(self, scripted):
        self.messages = FakeMessages(scripted)


def _make_tools(rng=None):
    rng = rng or np.random.default_rng(0)
    return AgentTools(
        model=GPSurrogate(),
        seen=[],
        y_train=[],
        rng=rng,
        target_tc_k=320.0,
    )


def test_serialize_roundtrip():
    c = Candidate(composition=(("La", 0.15), ("H", 0.85)), pressure_gpa=200.0)
    payload = _serialize_candidate(c)
    parsed = _parse_candidate(payload)
    assert parsed.composition == c.composition
    assert parsed.pressure_gpa == c.pressure_gpa


def test_parse_candidate_normalises_fractions():
    payload = {
        "composition": [["La", 0.10], ["H", 0.80]],
        "pressure_gpa": 200.0,
    }
    parsed = _parse_candidate(payload)
    assert abs(sum(f for _, f in parsed.composition) - 1.0) < 1e-9


def test_parse_candidate_rejects_unknown_element():
    with pytest.raises(ValueError, match="unknown element"):
        _parse_candidate(
            {"composition": [["Xx", 0.5], ["H", 0.5]], "pressure_gpa": 100}
        )


def test_parse_candidate_clips_pressure():
    parsed = _parse_candidate(
        {"composition": [["La", 0.15], ["H", 0.85]], "pressure_gpa": 9999.0}
    )
    assert parsed.pressure_gpa == 595.0


def test_tools_propose_random_pool_returns_valid():
    tools = _make_tools()
    out = tools.execute("propose_random_pool", {"n": 3})
    assert out["count"] == 3
    for c in out["candidates"]:
        parsed = _parse_candidate(c)
        assert symbolic_check(parsed).ok


def test_tools_predict_tc_warmup_then_trained():
    tools = _make_tools()
    payload = {"composition": [["La", 0.15], ["H", 0.85]], "pressure_gpa": 200.0}
    res = tools.execute("predict_tc", payload)
    assert res["warmup"] is True

    rng = np.random.default_rng(0)
    candidates = [sample_random(rng) for _ in range(8)]
    X = np.stack([featurize(c) for c in candidates])
    y = np.array([100.0 + i * 5 for i in range(len(candidates))])
    tools.model.fit(X, y)
    res = tools.execute("predict_tc", payload)
    assert "mean_tc_k" in res
    assert "std_tc_k" in res


def test_tools_inspect_history():
    tools = _make_tools()
    tools.seen = [Candidate(composition=(("La", 0.15), ("H", 0.85)), pressure_gpa=200.0)]
    tools.y_train = [180.5]
    out = tools.execute("inspect_history", {"last_n": 5})
    assert out["best_tc_k"] == 180.5
    assert out["total_measurements"] == 1
    assert out["rows"][0]["tc_k"] == 180.5


def test_agent_returns_candidate_on_submit():
    """Agent immediately calls submit_to_lab; propose returns the candidate."""
    payload = {
        "composition": [["La", 0.15], ["H", 0.85]],
        "pressure_gpa": 200.0,
    }
    client = FakeClient([_response([_tool_use("submit_to_lab", payload, "t1")])])
    agent = LLMHypothesizer(client=client, max_turns=5)
    tools = _make_tools()
    result = agent.propose(tools, round_idx=0, total_rounds=10)
    assert result is not None
    assert result.composition[0] == ("La", 0.15)
    assert result.pressure_gpa == 200.0


def test_agent_recovers_from_invalid_submit():
    """Bad submit returns an error tool_result; second submit succeeds."""
    bad = {"composition": [["Xx", 1.0]], "pressure_gpa": 200.0}
    good = {"composition": [["La", 0.15], ["H", 0.85]], "pressure_gpa": 200.0}
    client = FakeClient([
        _response([_tool_use("submit_to_lab", bad, "t1")]),
        _response([_tool_use("submit_to_lab", good, "t2")]),
    ])
    agent = LLMHypothesizer(client=client, max_turns=5)
    result = agent.propose(_make_tools(), round_idx=0, total_rounds=10)
    assert result is not None
    assert ("La", 0.15) in result.composition
    # Second call's messages must include the error tool_result from the first.
    second_messages = client.messages.calls[1]["messages"]
    assert any(
        isinstance(m["content"], list)
        and any(b.get("is_error") for b in m["content"] if isinstance(b, dict))
        for m in second_messages
    )


def test_agent_uses_intermediate_tools_then_submits():
    """Agent calls a non-terminal tool, gets a result, then submits."""
    inspect = _tool_use("propose_random_pool", {"n": 2}, "t1")
    submit = _tool_use(
        "submit_to_lab",
        {"composition": [["La", 0.15], ["H", 0.85]], "pressure_gpa": 200.0},
        "t2",
    )
    client = FakeClient([_response([inspect]), _response([submit])])
    agent = LLMHypothesizer(client=client, max_turns=5)
    result = agent.propose(_make_tools(), round_idx=0, total_rounds=10)
    assert result is not None
    # Two API calls: first for the pool, second for the submit.
    assert len(client.messages.calls) == 2


def test_agent_returns_none_when_no_submit():
    """Agent runs out of turns without submitting; propose returns None."""
    pool = _tool_use("propose_random_pool", {"n": 1}, "t1")
    client = FakeClient([_response([pool])] * 3)
    agent = LLMHypothesizer(client=client, max_turns=2)
    result = agent.propose(_make_tools(), round_idx=0, total_rounds=10)
    assert result is None


def test_agent_caches_system_prompt():
    """The system prompt sent to the client must carry cache_control."""
    payload = {"composition": [["La", 0.15], ["H", 0.85]], "pressure_gpa": 200.0}
    client = FakeClient([_response([_tool_use("submit_to_lab", payload, "t1")])])
    agent = LLMHypothesizer(client=client)
    agent.propose(_make_tools(), round_idx=0, total_rounds=10)
    call = client.messages.calls[0]
    system = call["system"]
    assert isinstance(system, list)
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert system[0]["text"] == SYSTEM_PROMPT


def test_tool_definitions_well_formed():
    """Every tool has the required schema fields."""
    names = {t["name"] for t in TOOLS}
    assert "submit_to_lab" in names
    assert "symbolic_check" in names
    for tool in TOOLS:
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"
