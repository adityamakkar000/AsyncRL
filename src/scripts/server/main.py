from __future__ import annotations

import logging
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from scripts.premption_tooling.main import Runtime, TPUJob, TPUType, Zone, run_session

logger = logging.getLogger(__name__)


class RunJobRequest(BaseModel):
    node_id: str
    zone: Zone
    tpu_type: TPUType
    runtime: Runtime
    cmd: str
    retries: int = 3
    name: str


class JobView(BaseModel):
    node_id: str
    name: str
    zone: str
    tpu_type: str
    runtime: str
    cmd: str
    retries_left: int
    tpu_status: str
    job_status: str


class JobQueueState:
    """Thread-safe job list + names; background worker runs run_session in a loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: list[TPUJob] = []
        self._name_by_node: dict[str, str] = {}
        self._shutdown = threading.Event()
        self._worker: threading.Thread | None = None

    def start_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._shutdown.clear()
        self._worker = threading.Thread(target=self._run_loop, name="tpu-run-session", daemon=True)
        self._worker.start()

    def shutdown(self) -> None:
        self._shutdown.set()
        if self._worker is not None:
            self._worker.join(timeout=30.0)

    def _run_loop(self) -> None:
        while not self._shutdown.is_set():
            with self._lock:
                pending = any(not j.is_job_finished for j in self._jobs)
            if not pending:
                time.sleep(0.25)
                continue
            try:
                with self._lock:
                    jobs_ref = self._jobs
                run_session(jobs_ref)
            except Exception:
                logger.exception("run_session crashed; retrying after delay")
                time.sleep(2.0)

    def add_job(self, body: RunJobRequest) -> None:
        if body.node_id in self._name_by_node:
            raise ValueError(f"node_id already queued: {body.node_id}")
        job = TPUJob(
            node_id=body.node_id,
            zone=body.zone,
            tpu_type=body.tpu_type,
            runtime=body.runtime,
            cmd=body.cmd,
            retries=body.retries,
        )
        log_root = Path("logs") / body.name
        log_root.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._jobs.append(job)
            self._name_by_node[body.node_id] = body.name

    def list_jobs(self) -> list[JobView]:
        with self._lock:
            snapshot = list(self._jobs)
            names = dict(self._name_by_node)
        out: list[JobView] = []
        for j in snapshot:
            name = names.get(j.node_id, j.node_id)
            out.append(
                JobView(
                    node_id=j.node_id,
                    name=name,
                    zone=j.zone.value,
                    tpu_type=j.tpu_type.value,
                    runtime=j.runtime.value,
                    cmd=j.cmd,
                    retries_left=j.retries,
                    tpu_status=j.tpu_status,
                    job_status=j.job_status,
                )
            )
        return out

    def delete_job(self, job_id: str) -> bool:
        """Remove job by node_id; terminate process and release TPU if allocated."""
        with self._lock:
            idx = next((i for i, j in enumerate(self._jobs) if j.node_id == job_id), None)
            if idx is None:
                return False
            job = self._jobs.pop(idx)
            self._name_by_node.pop(job_id, None)
        try:
            job.delete_tpu()
        except Exception:
            logger.exception("delete_tpu failed for %s", job_id)
        return True


state: JobQueueState | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global state
    logging.basicConfig(level=logging.INFO)
    state = JobQueueState()
    state.start_worker()
    yield
    if state is not None:
        state.shutdown()


app = FastAPI(title="TPU job queue", lifespan=lifespan)


def _get_state() -> JobQueueState:
    if state is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    return state


@app.post("/run_job")
def run_job(req: RunJobRequest) -> dict[str, Any]:
    try:
        _get_state().add_job(req)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return {"ok": True, "node_id": req.node_id, "name": req.name}


@app.get("/jobs", response_model=list[JobView])
def get_jobs() -> list[JobView]:
    return _get_state().list_jobs()


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, Any]:
    if not _get_state().delete_job(job_id):
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
    return {"ok": True, "deleted": job_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("scripts.server.main:app", host="0.0.0.0", port=8000, reload=False)