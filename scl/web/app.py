"""FastAPI app: REST + SSE for the discovery loop, plus static frontend."""

from __future__ import annotations

import asyncio
import csv
import io
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .runner import RunManager
from .storage import RunStore


_STATIC_DIR = Path(__file__).parent / "static"


class RunConfig(BaseModel):
    rounds: int = Field(30, ge=1, le=500)
    seed: int = 42
    pool_size: int = Field(200, ge=10, le=2000)
    init_size: int = Field(5, ge=1, le=50)
    kappa: float = Field(2.0, ge=0.0, le=10.0)
    falsify_every: int = Field(5, ge=0, le=100)
    inverse_every: int = Field(7, ge=0, le=100)
    nnqs_every: int = Field(6, ge=0, le=100)
    manifold_weight: float = Field(0.5, ge=0.0, le=10.0)
    target_tc_k: float = Field(320.0, ge=10.0, le=1000.0)
    use_agent: bool = False
    agent_model: str = "claude-opus-4-7"
    agent_effort: str = "xhigh"
    compare_baseline: bool = False


def create_app(runs_dir: Path | str = "runs") -> FastAPI:
    store = RunStore(Path(runs_dir))
    manager = RunManager(store)
    app = FastAPI(title="super-conductor-lab")
    app.state.manager = manager

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.post("/api/runs")
    def create_run(config: RunConfig) -> dict[str, Any]:
        cfg = config.model_dump(exclude={"compare_baseline"})
        run = manager.create(cfg, paired_baseline=config.compare_baseline)
        return {
            "run_id": run.id,
            "baseline_id": run.paired_baseline_id,
        }

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        return manager.list()

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        run = manager.get(run_id)
        if run is None:
            raise HTTPException(404, "no such run")
        return {
            "id": run.id,
            "config": run.config,
            "created_at": run.created_at,
            "status": run.status,
            "paired_baseline_id": run.paired_baseline_id,
            "events": run.events,
            "summary": run.summary,
            "error": run.error,
        }

    @app.get("/api/runs/{run_id}/export.json")
    def export_run_json(run_id: str) -> Response:
        run = manager.get(run_id)
        if run is None:
            raise HTTPException(404, "no such run")
        payload = {
            "id": run.id,
            "config": run.config,
            "created_at": run.created_at,
            "status": run.status,
            "paired_baseline_id": run.paired_baseline_id,
            "events": run.events,
            "summary": run.summary,
        }
        return Response(
            content=json.dumps(payload, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{run.id}.json"'},
        )

    @app.get("/api/runs/{run_id}/export.csv")
    def export_run_csv(run_id: str) -> Response:
        run = manager.get(run_id)
        if run is None:
            raise HTTPException(404, "no such run")
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow([
            "round", "success", "predicted_mean", "predicted_std",
            "measured_tc_k", "best_so_far_k", "quantum_proxy",
            "requested_formula", "requested_pressure_gpa",
            "realized_formula", "realized_pressure_gpa",
            "note",
        ])
        for ev in run.events:
            if ev.get("type") != "round":
                continue
            cand = ev.get("candidate") or {}
            real = ev.get("realized") or {}
            w.writerow([
                ev.get("round"),
                ev.get("success"),
                ev.get("predicted_mean"),
                ev.get("predicted_std"),
                ev.get("measured_tc_k"),
                ev.get("best_so_far_k"),
                ev.get("quantum_proxy"),
                cand.get("formula"),
                cand.get("pressure_gpa"),
                real.get("formula"),
                real.get("pressure_gpa"),
                ev.get("note"),
            ])
        return Response(
            content=buf.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{run.id}.csv"'},
        )

    @app.get("/api/runs/{run_id}/stream")
    async def stream(run_id: str) -> StreamingResponse:
        run = manager.get(run_id)
        if run is None:
            raise HTTPException(404, "no such run")

        async def event_gen():
            cursor = 0
            while True:
                # Snapshot to avoid mid-iteration mutation.
                snapshot = list(run.events)
                if cursor < len(snapshot):
                    for ev in snapshot[cursor:]:
                        yield f"data: {json.dumps(ev)}\n\n"
                    cursor = len(snapshot)
                if run.status not in ("running", "pending"):
                    # One last drain in case the executor appended after our snapshot.
                    snapshot = list(run.events)
                    for ev in snapshot[cursor:]:
                        yield f"data: {json.dumps(ev)}\n\n"
                    yield f"data: {json.dumps({'type': 'closed', 'status': run.status})}\n\n"
                    return
                await asyncio.sleep(0.1)

        return StreamingResponse(event_gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    return app
