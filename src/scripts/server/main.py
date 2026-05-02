from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Any
import shutil
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from scripts.premption_tooling.main import Runtime, TPUJob, TPUType, Zone, run_session

logger = logging.getLogger(__name__)


class RunJobRequest(BaseModel):
    node_id: str
    zone: Zone
    tpu_type: TPUType
    runtime: Runtime
    cmd: str
    retries: int = 3


class JobView(BaseModel):
    node_id: str
    zone: str
    tpu_type: str
    runtime: str
    cmd: str
    retries_left: int
    tpu_status: str
    job_status: str


class Server:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jobs: list[TPUJob] = []
        self.shutdown = threading.Event()
        self.worker: threading.Thread | None = None
        self.delete_queue: list[str] = []
        self.delete_lock = threading.Lock()

    def start_worker(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        self.shutdown.clear()
        self.worker = threading.Thread(target=self.run_loop, name="tpu-run-session", daemon=True)
        self.worker.start()

    def shutdown_worker(self) -> None:
        self.shutdown.set()
        if self.worker is not None:
            self.worker.join(timeout=30.0)

    def run_loop(self) -> None:
        while not self.shutdown.is_set():
            with self.lock:
                jobs_snapshot = list(self.jobs)
            with self.delete_lock:
                to_delete = list(self.delete_queue)
            pending = [j for j in jobs_snapshot if not j.is_job_finished_without_error]
            not_pending = [j for j in jobs_snapshot if j.is_job_finished_without_error]
            for j in not_pending:
                try:
                    self.delete_job(j.node_id)
                    logger.info("deleted job %s", j.node_id)
                except Exception:
                    logger.exception("delete_tpu failed for %s", j.node_id)
            for j in to_delete:
                try:
                    if self.delete_job(j):
                        logger.info("deleted job %s", j)
                    else: 
                        logger.info("failed to delete job %s, must send DELETE request again", j)
                    with self.delete_lock:
                        self.delete_queue.remove(j)
                except Exception:
                    logger.exception("delete_tpu failed for %s", j)
            if len(pending) == 0:
                time.sleep(1)
                continue
            try:
                with self.lock:
                    jobs_ref = list(self.jobs)
                run_session(jobs_ref)
            except Exception:
                logger.exception("run_session crashed; retrying after delay")
            finally:
                time.sleep(1)

    def add_job(self, body: RunJobRequest) -> None:
        with self.lock:
            if body.node_id in [j.node_id for j in self.jobs]:
                raise ValueError(f"node_id already queued: {body.node_id}")
            job = TPUJob(
                node_id=body.node_id,
                zone=body.zone,
                tpu_type=body.tpu_type,
                runtime=body.runtime,
                cmd=body.cmd,
                retries=body.retries,
            )
            self.jobs.append(job)

    def list_jobs(self) -> list[JobView]:
        with self.lock:
            snapshot = list(self.jobs)
        out: list[JobView] = []
        for j in snapshot:
            out.append(
                JobView(
                    node_id=j.node_id,
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
        with self.lock:
            idx = next((i for i, j in enumerate(self.jobs) if j.node_id == job_id), None)
            if idx is None:
                return False
            job = self.jobs[idx]
            del self.jobs[idx]
        try:
            job.delete_tpu()
        except Exception:
            logger.exception("delete_tpu failed for %s, adding back job to the queue", job_id)
            with self.lock:
                self.jobs.append(job)
            return False
        shutil.rmtree(f"{job.home_dir}/{job.node_id}")
        log_path = f"{job.home_dir}/logs/{job.node_id}.txt"
        if os.path.exists(log_path):
            os.remove(log_path)
        return True


state: Server | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global state
    logging.basicConfig(level=logging.INFO)
    state = Server()
    state.start_worker()
    yield
    if state is not None:
        state.shutdown_worker()


app = FastAPI(title="TPU Server (Mac Mini)", lifespan=lifespan)


def get_state() -> Server:
    if state is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    return state


@app.post("/run_job")
def run_job(req: RunJobRequest) -> dict[str, Any]:
    try:
        get_state().add_job(req)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return {"ok": True, "node_id": req.node_id}


@app.get("/jobs", response_model=list[JobView])
def get_jobs() -> list[JobView]:
    return get_state().list_jobs()


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, Any]:
    with get_state().delete_lock:
        if job_id in get_state().delete_queue:
            return {"ok": False, "deleted": job_id}
        get_state().delete_queue.append(job_id)

    return {"ok": True, "deleted": job_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("scripts.server.main:app", host="0.0.0.0", port=8000, reload=False)
