"""
Minimal in-process background job runner. The frontend wizard needs each pipeline step (LLM
extraction over dozens of samples, a SAASBO/NUTS fit) to run without blocking the HTTP request
past typical browser/proxy timeouts, and to show progress -- so every long operation is
started as a job and polled via GET /api/jobs/{id} instead of awaited inline.

Single-process, in-memory: fine for a local, single-user tool. Not meant to survive a server
restart or to scale beyond one machine.
"""
from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional


@dataclass
class Job:
    id: str
    kind: str
    status: str = "pending"          # pending | running | done | error
    progress: Optional[dict] = None
    result: Any = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, kind: str, fn: Callable[..., Any], *args, **kwargs) -> str:
        job_id = uuid.uuid4().hex
        job = Job(id=job_id, kind=kind, status="pending")
        with self._lock:
            self._jobs[job_id] = job

        def set_progress(p: dict):
            with self._lock:
                self._jobs[job_id].progress = p

        def run():
            with self._lock:
                self._jobs[job_id].status = "running"
            try:
                result = fn(*args, progress_cb=set_progress, **kwargs)
                with self._lock:
                    self._jobs[job_id].status = "done"
                    self._jobs[job_id].result = result
            except Exception as e:  # noqa: BLE001
                with self._lock:
                    self._jobs[job_id].status = "error"
                    self._jobs[job_id].error = f"{e}\n{traceback.format_exc(limit=6)}"

        threading.Thread(target=run, daemon=True).start()
        return job_id

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def to_dict(self, job_id: str) -> Optional[dict]:
        job = self.get(job_id)
        if job is None:
            return None
        return {"id": job.id, "kind": job.kind, "status": job.status, "progress": job.progress,
                "result": job.result, "error": job.error, "created_at": job.created_at}


jobs = JobManager()
