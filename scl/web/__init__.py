"""Web UI for super-conductor-lab.

FastAPI app + SSE round streaming + JSONL persistence + a single-page
vanilla-JS frontend. Optional dependency group: install with `pip install
-e '.[web]'` and start with `scl serve`.
"""

from .app import create_app
from .runner import Run, RunManager

__all__ = ["create_app", "Run", "RunManager"]
