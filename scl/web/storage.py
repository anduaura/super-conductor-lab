"""JSONL persistence for completed runs.

Each run is one file: `runs/<run_id>.jsonl`.
  Line 1: meta — id, config, created_at, status, paired baseline id (if any).
  Line 2..N: round events.
  Last line: summary — best Tc, best candidate, total successes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunRecord:
    id: str
    config: dict[str, Any]
    created_at: float
    status: str
    paired_baseline_id: str | None
    events: list[dict[str, Any]]
    summary: dict[str, Any] | None


class RunStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        return self.root / f"{run_id}.jsonl"

    def save(self, record: RunRecord) -> None:
        path = self._path(record.id)
        with path.open("w") as f:
            f.write(json.dumps({
                "type": "meta",
                "id": record.id,
                "config": record.config,
                "created_at": record.created_at,
                "status": record.status,
                "paired_baseline_id": record.paired_baseline_id,
            }) + "\n")
            for ev in record.events:
                f.write(json.dumps(ev) + "\n")
            if record.summary is not None:
                f.write(json.dumps({"type": "summary", **record.summary}) + "\n")

    def load(self, run_id: str) -> RunRecord | None:
        path = self._path(run_id)
        if not path.exists():
            return None
        meta: dict[str, Any] = {}
        events: list[dict[str, Any]] = []
        summary: dict[str, Any] | None = None
        with path.open() as f:
            for line in f:
                ev = json.loads(line)
                if ev.get("type") == "meta":
                    meta = ev
                elif ev.get("type") == "summary":
                    summary = {k: v for k, v in ev.items() if k != "type"}
                else:
                    events.append(ev)
        return RunRecord(
            id=meta.get("id", run_id),
            config=meta.get("config", {}),
            created_at=meta.get("created_at", 0.0),
            status=meta.get("status", "unknown"),
            paired_baseline_id=meta.get("paired_baseline_id"),
            events=events,
            summary=summary,
        )

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.jsonl"))

    def list_summaries(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for run_id in self.list_ids():
            rec = self.load(run_id)
            if rec is None:
                continue
            out.append({
                "id": rec.id,
                "created_at": rec.created_at,
                "status": rec.status,
                "paired_baseline_id": rec.paired_baseline_id,
                "best_tc_k": (rec.summary or {}).get("best_tc_k"),
                "rounds": (rec.summary or {}).get("rounds"),
                "successful_rounds": (rec.summary or {}).get("successful_rounds"),
            })
        out.sort(key=lambda r: r["created_at"], reverse=True)
        return out
