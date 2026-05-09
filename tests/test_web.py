"""Web-layer tests.

Skipped automatically when fastapi / httpx aren't installed (the web layer is
behind the [web] optional dep group).
"""

import json
import time
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from scl.web.app import create_app
from scl.web.runner import RunManager, serialize_round
from scl.web.storage import RunStore


_TINY_CONFIG = {
    "rounds": 3,
    "seed": 1,
    "pool_size": 20,
    "init_size": 2,
    "kappa": 2.0,
    "falsify_every": 0,
    "inverse_every": 0,
    "nnqs_every": 0,
    "manifold_weight": 0.0,
    "target_tc_k": 320.0,
}


def _wait_done(client: TestClient, run_id: str, timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/runs/{run_id}")
        assert r.status_code == 200
        data = r.json()
        if data["status"] in ("done", "failed"):
            return data
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish within {timeout}s")


def test_run_lifecycle_and_persistence(tmp_path: Path):
    app = create_app(runs_dir=tmp_path)
    with TestClient(app) as client:
        r = client.post("/api/runs", json={**_TINY_CONFIG, "compare_baseline": False})
        assert r.status_code == 200
        run_id = r.json()["run_id"]
        assert r.json()["baseline_id"] is None

        data = _wait_done(client, run_id)
        assert data["status"] == "done"
        assert data["summary"] is not None
        # Summary rounds count includes the cold-start seeds.
        assert data["summary"]["rounds"] == _TINY_CONFIG["rounds"] + _TINY_CONFIG["init_size"]
        # Persisted to disk.
        assert (tmp_path / f"{run_id}.jsonl").exists()


def test_paired_baseline(tmp_path: Path):
    app = create_app(runs_dir=tmp_path)
    with TestClient(app) as client:
        r = client.post("/api/runs", json={**_TINY_CONFIG, "compare_baseline": True})
        assert r.status_code == 200
        body = r.json()
        assert body["baseline_id"] is not None
        active = _wait_done(client, body["run_id"])
        baseline = _wait_done(client, body["baseline_id"])
        assert active["status"] == "done"
        assert baseline["status"] == "done"
        assert active["paired_baseline_id"] == body["baseline_id"]


def test_history_lists_persisted_run(tmp_path: Path):
    app = create_app(runs_dir=tmp_path)
    with TestClient(app) as client:
        r = client.post("/api/runs", json={**_TINY_CONFIG, "compare_baseline": False})
        run_id = r.json()["run_id"]
        _wait_done(client, run_id)
        listing = client.get("/api/runs").json()
        assert any(item["id"] == run_id for item in listing)


def test_sse_stream_completes(tmp_path: Path):
    app = create_app(runs_dir=tmp_path)
    with TestClient(app) as client:
        run_id = client.post("/api/runs", json={**_TINY_CONFIG}).json()["run_id"]
        # Drain the stream — should end with a 'closed' frame.
        with client.stream("GET", f"/api/runs/{run_id}/stream") as resp:
            assert resp.status_code == 200
            saw_round = False
            saw_close = False
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line[len("data: "):])
                if payload.get("type") == "round":
                    saw_round = True
                if payload.get("type") == "closed":
                    saw_close = True
                    break
            assert saw_round and saw_close


def test_sync_runner_produces_summary(tmp_path: Path):
    """Direct RunManager test with sync=True for deterministic CI behavior."""
    store = RunStore(tmp_path)
    mgr = RunManager(store)
    run = mgr.create(_TINY_CONFIG, paired_baseline=False, sync=True)
    assert run.status == "done"
    assert run.summary is not None
    assert run.summary["rounds"] == _TINY_CONFIG["rounds"] + _TINY_CONFIG["init_size"]
    # Reload from disk.
    rec = store.load(run.id)
    assert rec is not None
    assert rec.summary == run.summary


def test_serialize_round_shape():
    """Smoke: serialize_round produces JSON-friendly output for a real RoundLog."""
    from scl.candidates import Candidate
    from scl.loop import RoundLog
    c = Candidate(composition=(("La", 0.15), ("H", 0.85)), pressure_gpa=200.0)
    rl = RoundLog(
        round=0, candidate=c, realized=c,
        predicted_mean=180.0, predicted_std=15.0, quantum_proxy=-1.2,
        measured_tc_k=170.0, success=True, note="UCB", best_so_far_k=170.0,
    )
    payload = serialize_round(rl)
    json.dumps(payload)  # must be JSON serializable
    assert payload["candidate"]["formula"].startswith("La")
