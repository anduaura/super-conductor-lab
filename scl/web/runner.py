"""Run lifecycle + event fan-out.

A `Run` owns an event list and a status. The `RunManager` spawns each run in
its own thread so the event loop (and other concurrent runs) stay responsive,
and persists the completed run as JSONL.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..candidates import Candidate
from ..loop import LoopResult, RoundLog, run_loop
from .storage import RunRecord, RunStore


def serialize_candidate(c: Candidate) -> dict[str, Any]:
    return {
        "composition": [[e, float(f)] for e, f in c.composition],
        "pressure_gpa": float(c.pressure_gpa),
        "formula": c.formula(),
    }


def serialize_round(rl: RoundLog) -> dict[str, Any]:
    return {
        "type": "round",
        "round": rl.round,
        "candidate": serialize_candidate(rl.candidate),
        "realized": serialize_candidate(rl.realized),
        "predicted_mean": rl.predicted_mean,
        "predicted_std": rl.predicted_std,
        "quantum_proxy": rl.quantum_proxy,
        "measured_tc_k": rl.measured_tc_k,
        "success": rl.success,
        "note": rl.note,
        "best_so_far_k": rl.best_so_far_k,
    }


def serialize_summary(result: LoopResult) -> dict[str, Any]:
    successes = [r for r in result.rounds if r.success]
    return {
        "rounds": len(result.rounds),
        "successful_rounds": len(successes),
        "best_tc_k": result.best_tc_k,
        "best_candidate": (
            serialize_candidate(result.best_candidate)
            if result.best_candidate is not None else None
        ),
    }


@dataclass
class Run:
    id: str
    config: dict[str, Any]
    created_at: float
    paired_baseline_id: str | None = None
    status: str = "pending"           # pending | running | done | failed
    events: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] | None = None
    error: str | None = None

    def append_event(self, event: dict[str, Any]) -> None:
        self.events.append(event)


class RunManager:
    """Spawns runs in background threads, persists them, lists past runs."""

    def __init__(self, store: RunStore):
        self.store = store
        self.runs: dict[str, Run] = {}
        self._lock = threading.Lock()

    def get(self, run_id: str) -> Run | None:
        with self._lock:
            run = self.runs.get(run_id)
        if run is not None:
            return run
        rec = self.store.load(run_id)
        if rec is None:
            return None
        run = Run(
            id=rec.id,
            config=rec.config,
            created_at=rec.created_at,
            paired_baseline_id=rec.paired_baseline_id,
            status=rec.status,
            events=rec.events,
            summary=rec.summary,
        )
        with self._lock:
            self.runs[run.id] = run
        return run

    def list(self) -> list[dict[str, Any]]:
        live = []
        with self._lock:
            for run in self.runs.values():
                live.append({
                    "id": run.id,
                    "created_at": run.created_at,
                    "status": run.status,
                    "paired_baseline_id": run.paired_baseline_id,
                    "best_tc_k": (run.summary or {}).get("best_tc_k"),
                    "rounds": (run.summary or {}).get("rounds"),
                    "successful_rounds": (run.summary or {}).get("successful_rounds"),
                })
        # Merge with persisted runs (live takes precedence).
        seen = {r["id"] for r in live}
        for s in self.store.list_summaries():
            if s["id"] not in seen:
                live.append(s)
        live.sort(key=lambda r: r["created_at"] or 0, reverse=True)
        return live

    def create(self, config: dict[str, Any], paired_baseline: bool = False,
               sync: bool = False) -> Run:
        run = Run(id=uuid.uuid4().hex[:8], config=dict(config),
                  created_at=time.time())

        baseline: Run | None = None
        if paired_baseline:
            baseline_config = dict(config)
            baseline_config.update(
                random_select_only=True,
                falsify_every=0,
                inverse_every=0,
                nnqs_every=0,
                manifold_weight=0.0,
            )
            baseline = Run(
                id=uuid.uuid4().hex[:8],
                config=baseline_config,
                created_at=time.time(),
            )
            run.paired_baseline_id = baseline.id

        with self._lock:
            self.runs[run.id] = run
            if baseline is not None:
                self.runs[baseline.id] = baseline

        if sync:
            self._execute(run)
            if baseline is not None:
                self._execute(baseline)
        else:
            threading.Thread(target=self._execute, args=(run,),
                             daemon=True).start()
            if baseline is not None:
                threading.Thread(target=self._execute, args=(baseline,),
                                 daemon=True).start()
        return run

    def _execute(self, run: Run) -> None:
        run.status = "running"
        try:
            def cb(rl: RoundLog) -> None:
                run.append_event(serialize_round(rl))
            result = run_loop(on_round=cb, **run.config)
            run.summary = serialize_summary(result)
            run.append_event({"type": "summary", **run.summary})
            run.status = "done"
        except Exception as e:
            run.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            run.append_event({"type": "error", "message": run.error})
            run.status = "failed"
        finally:
            self.store.save(RunRecord(
                id=run.id, config=run.config, created_at=run.created_at,
                status=run.status, paired_baseline_id=run.paired_baseline_id,
                events=[e for e in run.events if e.get("type") != "summary"],
                summary=run.summary,
            ))
