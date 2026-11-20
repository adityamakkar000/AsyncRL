from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import re
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from src.scripts.launch.tpu import Runtime, TPUType, Zone
from src.scripts.server.job import TPUJob

logger = logging.getLogger(__name__)

_SAFE_NODE_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")
STATE_FILE = os.path.expanduser(os.getenv("TPU_SERVER_STATE", "~/.tpu_server/jobs.json"))
TERMINAL_TTL_SECONDS = 24 * 3600


def read_ui(name: str) -> str:
    with open(os.path.join(UI_DIR, name), encoding="utf-8") as f:
        return f.read()


class RunJobRequest(BaseModel):
    node_id: str
    zone: Zone
    tpu_type: TPUType
    runtime: Runtime
    cmd: str
    retries: int = 3
    launched_by: str = ""
    cwd: str = ""


class JobView(BaseModel):
    node_id: str
    state: str
    job_status: str
    tpu_status: str
    message: str
    zone: str
    tpu_type: str
    runtime: str
    cmd: str
    retries_left: int
    preemptions: int
    attempts: int
    launched_by: str
    created_at: float
    state_since: float
    finished_at: float | None


class JobDetail(JobView):
    events: list[list[Any]]
    log_dir: str


def view(j: TPUJob) -> dict[str, Any]:
    return {
        "node_id": j.node_id,
        "state": j.state.value,
        "job_status": j.state.value,
        "tpu_status": j.tpu_status,
        "message": j.message,
        "zone": j.zone.value,
        "tpu_type": j.tpu_type.value,
        "runtime": j.runtime.value,
        "cmd": j.cmd,
        "retries_left": j.retries,
        "preemptions": j.preemptions,
        "attempts": j.attempts,
        "launched_by": j.launched_by,
        "created_at": j.created_at,
        "state_since": j.state_since,
        "finished_at": j.finished_at,
    }


class Server:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.jobs: dict[str, TPUJob] = {}
        self.shutdown = threading.Event()
        self.janitor: threading.Thread | None = None

    def start(self) -> None:
        self.load()
        for job in list(self.jobs.values()):
            job.start()
        self.janitor = threading.Thread(target=self.janitor_loop, name="janitor", daemon=True)
        self.janitor.start()

    def stop(self) -> None:
        self.shutdown.set()
        with self.lock:
            jobs = list(self.jobs.values())
        for job in jobs:
            job.halt()
        self.save()

    def load(self) -> None:
        if not os.path.exists(STATE_FILE):
            return
        with open(STATE_FILE) as f:
            records = json.load(f)
        for r in records:
            try:
                job = TPUJob.from_record(r, on_change=self.on_change)
            except Exception:
                logger.exception("could not restore job %s", r.get("node_id"))
                continue
            self.jobs[job.node_id] = job
        logger.info("restored %d jobs from %s", len(self.jobs), STATE_FILE)

    def save(self) -> None:
        with self.lock:
            records = [j.to_record() for j in self.jobs.values()]
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
            tmp = f"{STATE_FILE}.tmp"
            with open(tmp, "w") as f:
                json.dump(records, f)
            os.replace(tmp, STATE_FILE)

    def on_change(self, _job: TPUJob | None) -> None:
        try:
            self.save()
        except Exception:
            logger.exception("failed to save server state")

    def janitor_loop(self) -> None:
        while not self.shutdown.wait(60):
            now = time.time()
            with self.lock:
                expired = [
                    k
                    for k, j in self.jobs.items()
                    if j.is_terminal and j.finished_at and now - j.finished_at > TERMINAL_TTL_SECONDS
                ]
                for k in expired:
                    del self.jobs[k]
            if expired:
                logger.info("pruned finished jobs: %s", expired)
                self.on_change(None)

    def add_job(self, body: RunJobRequest) -> bool:
        if not _SAFE_NODE_ID.match(body.node_id):
            raise ValueError(f"invalid node_id: {body.node_id}")
        job = TPUJob(
            node_id=body.node_id,
            zone=body.zone,
            tpu_type=body.tpu_type,
            runtime=body.runtime,
            cmd=body.cmd,
            cwd=body.cwd,
            retries=body.retries,
            launched_by=body.launched_by,
            on_change=self.on_change,
        )
        with self.lock:
            old = self.jobs.get(body.node_id)
            if old is not None and not old.is_terminal:
                raise KeyError(f"{body.node_id} is already {old.state.value}")
            self.jobs[body.node_id] = job
        job.event(
            f"submitted by {body.launched_by or 'unknown'}" + (f" (replaces {old.state.value} run)" if old else "")
        )
        job.start()
        return old is not None

    def get(self, node_id: str) -> TPUJob:
        with self.lock:
            job = self.jobs.get(node_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {node_id}")
        return job

    def list_jobs(self) -> list[dict[str, Any]]:
        with self.lock:
            snapshot = list(self.jobs.values())
        return [view(j) for j in sorted(snapshot, key=lambda j: (j.created_at, j.node_id))]

    def delete_job(self, node_id: str, force: bool = False) -> str:
        with self.lock:
            job = self.jobs.get(node_id)
            if job is None:
                raise HTTPException(status_code=404, detail=f"no job {node_id}")
            was_terminal = job.is_terminal
            if was_terminal or force:
                del self.jobs[node_id]
        if not was_terminal and not force:
            job.cancel()
            return "cancelling"
        if not was_terminal:
            job.on_change = None
            job.force_remove()
        self.on_change(None)
        return "removed" if was_terminal else "force-removed, TPU may still exist"


state: Server | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global state
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    state = Server()
    state.start()
    yield
    state.stop()


app = FastAPI(title="TPU Server", lifespan=lifespan)


def get_state() -> Server:
    if state is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    return state


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return read_ui("index.html")


@app.post("/run_job")
def run_job(req: RunJobRequest) -> dict[str, Any]:
    try:
        replaced = get_state().add_job(req)
    except KeyError as e:
        raise HTTPException(status_code=409, detail=str(e.args[0])) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"ok": True, "node_id": req.node_id, "replaced": replaced}


@app.get("/code/{name}")
def code_exists(name: str) -> dict[str, Any]:
    if not _SAFE_NODE_ID.match(name):
        raise HTTPException(status_code=400, detail="invalid name")
    return {"exists": os.path.isdir(os.path.join(os.path.expanduser("~"), "jobs", name))}


@app.get("/jobs", response_model=list[JobView])
def get_jobs() -> list[dict[str, Any]]:
    return get_state().list_jobs()


@app.get("/jobs/{node_id}", response_model=JobDetail)
def get_job(node_id: str) -> dict[str, Any]:
    job = get_state().get(node_id)
    return {**view(job), "events": list(job.events), "log_dir": job.log_dir}


@app.delete("/jobs/{node_id}")
def delete_job(node_id: str, force: bool = False) -> dict[str, Any]:
    return {"ok": True, "node_id": node_id, "action": get_state().delete_job(node_id, force)}


def log_dir_for_node(node_id: str) -> str:
    if not _SAFE_NODE_ID.match(node_id):
        raise HTTPException(status_code=400, detail="invalid node_id")
    with get_state().lock:
        job = get_state().jobs.get(node_id)
    if job is not None:
        return job.log_dir
    matches = glob.glob(os.path.join(os.path.expanduser("~"), "jobs", "*", "logs", node_id))
    if not matches:
        raise HTTPException(status_code=404, detail=f"no logs for {node_id}")
    return max(matches, key=os.path.getmtime)


async def _wait_for_log_file(path: str, timeout_s: float = 120.0, poll_s: float = 0.25) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if os.path.isfile(path):
            return
        await asyncio.sleep(poll_s)
    raise HTTPException(status_code=404, detail=f"log file not found: {path}")


def stream_file(path: str, tail_bytes: int | None = None) -> StreamingResponse:
    async def lines() -> AsyncIterator[bytes]:
        with open(path, encoding="utf-8", errors="replace") as f:
            if tail_bytes is not None:
                size = os.path.getsize(path)
                if size > tail_bytes:
                    f.seek(size - tail_bytes)
                    f.readline()
            while True:
                line = await asyncio.to_thread(f.readline)
                if line:
                    yield line.encode("utf-8")
                else:
                    await asyncio.sleep(0.2)

    return StreamingResponse(lines(), media_type="text/plain; charset=utf-8")


@app.get("/logs/{node_id}/stream")
async def stream_job_log(node_id: str, tail_kb: int | None = None) -> StreamingResponse:
    path = os.path.join(log_dir_for_node(node_id), "log.txt")
    await _wait_for_log_file(path)
    return stream_file(path, None if tail_kb is None else tail_kb * 1024)


MAX_CHUNK_BYTES = 1024 * 1024


@app.get("/logs/{node_id}/chunk")
def log_chunk(
    node_id: str, rank: int | None = None, before: int | None = None, after: int | None = None, size: int = 65536
) -> dict[str, Any]:
    log_dir = log_dir_for_node(node_id)
    path = os.path.join(log_dir, "log.txt") if rank is None else os.path.join(log_dir, "ranks", f"w{rank}.txt")
    if not os.path.isfile(path):
        return {"start": 0, "end": 0, "total": 0, "text": "", "missing": True}
    total = os.path.getsize(path)
    size = max(1, min(size, MAX_CHUNK_BYTES))
    if after is not None:
        start = min(max(after, 0), total)
        end = min(total, start + size)
    else:
        end = total if before is None else min(max(before, 0), total)
        start = max(0, end - size)
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
    if after is None and start > 0 and b"\n" in data:
        cut = data.index(b"\n") + 1
        data, start = data[cut:], start + cut
    if after is not None and not data.endswith(b"\n"):
        if b"\n" in data:
            keep = data.rindex(b"\n") + 1
            data, end = data[:keep], start + keep
        elif len(data) < size:
            data, end = b"", start
    return {
        "start": start,
        "end": end,
        "total": total,
        "text": data.decode("utf-8", errors="replace"),
        "missing": False,
    }


@app.get("/view/{node_id}", response_class=HTMLResponse)
def view_log(node_id: str) -> str:
    log_dir_for_node(node_id)
    return read_ui("view.html")


@app.get("/logs/{node_id}/ranks")
def list_rank_logs(node_id: str) -> list[str]:
    ranks_dir = os.path.join(log_dir_for_node(node_id), "ranks")
    if not os.path.isdir(ranks_dir):
        return []
    return sorted(os.listdir(ranks_dir), key=lambda f: int(f[1:-4]) if f[1:-4].isdigit() else -1)


@app.get("/logs/{node_id}/ranks/{index}/stream")
async def stream_rank_log(node_id: str, index: int, tail_kb: int | None = None) -> StreamingResponse:
    path = os.path.join(log_dir_for_node(node_id), "ranks", f"w{index}.txt")
    await _wait_for_log_file(path)
    return stream_file(path, None if tail_kb is None else tail_kb * 1024)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.scripts.server.main:app", host="0.0.0.0", port=8000, reload=False)
