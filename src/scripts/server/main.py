from __future__ import annotations

import asyncio
import logging
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.scripts.premption_tooling.main import Runtime, TPUJob, TPUType, Zone, run_session

logger = logging.getLogger(__name__)


class RunJobRequest(BaseModel):
    node_id: str
    zone: Zone
    tpu_type: TPUType
    runtime: Runtime
    cmd: str
    retries: int = 3
    launched_by: str = ""
    cwd: str = ""
    keep_logs: bool = True


class JobView(BaseModel):
    node_id: str
    zone: str
    tpu_type: str
    runtime: str
    cmd: str
    retries_left: int
    tpu_status: str
    job_status: str
    launched_by: str


class Server:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jobs: dict[str, TPUJob] = {}  # Keying by node_id prevents accidental duplicates
        self.delete_queue: set[str] = set()
        self.shutdown = threading.Event()
        self.worker: threading.Thread | None = None
        self.executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix="tpu-worker")

    def start_worker(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        self.shutdown.clear()
        self.worker = threading.Thread(target=self.run_loop, name="tpu-main-loop", daemon=True)
        self.worker.start()

    def shutdown_worker(self) -> None:
        self.shutdown.set()
        if self.worker is not None:
            self.worker.join(timeout=30.0)
        self.executor.shutdown(wait=True)

    def run_loop(self) -> None:
        while not self.shutdown.is_set():
            start_time = time.time()

            with self.lock:
                for node_id, job in list(self.jobs.items()):
                    if job.is_job_finished_without_error:
                        self.delete_queue.add(node_id)

                jobs_snapshot = list(self.jobs.values())
                to_delete_snapshot = list(self.delete_queue)

            for node_id in to_delete_snapshot:
                try:
                    if self._internal_delete_job(node_id):
                        logger.info("Successfully deleted job assets for node: %s", node_id)
                        with self.lock:
                            self.delete_queue.discard(node_id)
                    else:
                        logger.warning("Job %s not found or busy; will retry deletion.", node_id)
                except Exception:
                    logger.exception("Involuntary background deletion failure for %s", node_id)

            try:
                run_session(jobs_snapshot, self.executor)
            except Exception:
                logger.exception("Critical control plane crash during run_session execution loop")

            elapsed = time.time() - start_time
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)

    def add_job(self, body: RunJobRequest) -> None:
        with self.lock:
            if body.node_id in self.jobs:
                raise ValueError(f"node_id already registered or active: {body.node_id}")

            self.jobs[body.node_id] = TPUJob(
                node_id=body.node_id,
                zone=body.zone,
                tpu_type=body.tpu_type,
                runtime=body.runtime,
                cmd=body.cmd,
                retries=body.retries,
                cwd=body.cwd,
                launched_by=body.launched_by,
                keep_logs=body.keep_logs,
            )

    def list_jobs(self) -> list[JobView]:
        with self.lock:
            snapshot = list(self.jobs.values())

        return [
            JobView(
                node_id=j.node_id,
                zone=j.zone.value,
                tpu_type=j.tpu_type.value,
                runtime=j.runtime.value,
                cmd=j.cmd,
                retries_left=j.retries,
                tpu_status=j.tpu_status,
                job_status=j.job_status,
                launched_by=j.launched_by,
            )
            for j in snapshot
        ]

    def _internal_delete_job(self, job_id: str) -> bool:
        """Internal worker method to isolate systemic asset teardown safely."""
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return False

        try:
            job.delete_tpu()
        except Exception:
            logger.exception("GCP Cloud resource deletion failed for %s; tracking retained.", job_id)
            return False

        try:
            shutil.rmtree(job.launch_dir, ignore_errors=True)
        except Exception:
            logger.exception("Failed to drop local launch directory for: %s", job.launch_dir)

        log_path = os.path.join(os.path.expanduser("~"), "logs", f"{job.node_id}.txt")
        if not job.keep_logs and os.path.exists(log_path):
            try:
                os.remove(log_path)
            except Exception:
                logger.exception("Failed deleting unneeded log target: %s", log_path)

        with self.lock:
            self.jobs.pop(job_id, None)
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


app = FastAPI(title="Distributed TPU Orchestration Engine", lifespan=lifespan)


def get_state() -> Server:
    if state is None:
        raise HTTPException(status_code=503, detail="Control plane state initializing")
    return state


@app.post("/run_job")
def run_job(req: RunJobRequest) -> dict[str, Any]:
    try:
        get_state().add_job(req)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True, "node_id": req.node_id}


@app.get("/jobs", response_model=list[JobView])
def get_jobs() -> list[JobView]:
    return get_state().list_jobs()


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, Any]:
    srv = get_state()
    with srv.lock:
        if job_id not in srv.jobs:
            raise HTTPException(status_code=404, detail="Job identity missing")
        srv.delete_queue.add(job_id)
    return {"ok": True, "deleted": job_id}


def _log_path_for_node(node_id: str) -> str:
    return os.path.join(os.path.expanduser("~"), "logs", f"{node_id}.txt")


@app.get("/logs/{node_id}/stream")
async def stream_job_log(node_id: str) -> StreamingResponse:
    path = _log_path_for_node(node_id)

    async def lines() -> AsyncIterator[bytes]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"--- Tail Stream Initialized for {node_id} (Awaiting TPU Node Activation) ---\n")

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            while True:
                line = await asyncio.to_thread(f.readline)
                if line:
                    yield line.encode("utf-8")
                else:
                    srv = get_state()
                    with srv.lock:
                        job = srv.jobs.get(node_id)

                    if job is None or job.is_job_finished:
                        final_line = await asyncio.to_thread(f.readline)
                        if final_line:
                            yield final_line.encode("utf-8")
                        break

                    await asyncio.sleep(0.5)

    return StreamingResponse(lines(), media_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.scripts.server.main:app", host="0.0.0.0", port=8000, reload=False)
