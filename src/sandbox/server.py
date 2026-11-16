import asyncio
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .sandbox import ExecResult, Sandbox, SandboxConfig


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"  
    port: int = 8100
    max_sandboxes: int = 64  # live boxes at once; past this, create returns 503
    idle_ttl: float = 30 * 60  # seconds without activity before the box is destroyed
    reap_interval: float = 10.0  # seconds between reaper passes
    max_timeout: float = 30 * 60  # upper bound on a single exec's timeout
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)  # defaults for every box



class CreateRequest(BaseModel):
    image: str | None = None  # None -> ServerConfig.sandbox.image


class CreateResponse(BaseModel):
    sandbox_id: str

class ExecRequest(BaseModel):
    command: str
    timeout: float = 30.0


class ExecResponse(BaseModel):
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool
    sandbox_died: bool
    duration: float
    stdout_truncated: bool
    stderr_truncated: bool


class HealthResponse(BaseModel):
    ok: bool  # True when docker is reachable and the configured runtime exists
    runtime: str | None
    live: int
    capacity: int


class SandboxNotFoundError(Exception):
    """Unknown or already-destroyed sandbox id -> 404."""


class SandboxCapacityError(Exception):
    """max_sandboxes reached -> 503, the client should retry later."""


@dataclass
class SandboxEntry:
    sandbox: Sandbox
    created_at: float
    last_used: float
    in_flight: int = 0  # execs running or waiting on `lock`; the reaper must never destroy a busy box
    lock: threading.Lock = field(default_factory=threading.Lock)  # one command at a time per box


class SandboxManager:
    """Thread-safe registry of live sandboxes. All methods are blocking (they call docker), so the
    endpoints run them in threads; `self.lock` guards `self.entries`, never hold it while calling docker."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.lock = threading.Lock()
        self.entries: dict[str, SandboxEntry] = {}
        
        ...  # TODO: create one shared docker client with make_client(), sized for max_sandboxes

    def startup(self):
        """Called once before serving: remove leftover boxes from a previous run (cleanup_all),
        check docker and the runtime are usable, and pull the default image so the first create is fast."""
        ...

    def shutdown(self):
        """Called once on exit: destroy every live box."""
        ...

    def create(self, image: str | None = None) -> str:
        """Reserve a slot (raise SandboxCapacityError if full), start a Sandbox, register it, return its id.
        Reserve the slot under the lock *before* starting the container, so concurrent creates can't
        overshoot max_sandboxes, and release it if the start fails."""
        ...

    def exec(self, sandbox_id: str, command: str, timeout: float) -> ExecResult:
        """Look up the box (SandboxNotFoundError if unknown), clamp timeout to max_timeout, run the command
        while holding the entry's `lock`, so a second command on the same box waits for the first.
        Bump in_flight / last_used around the call. If the box died: replace gVisor's internal stderr with a
        clean message and destroy + unregister the box, since it can't be used again."""
        ...

    def destroy(self, sandbox_id: str):
        """Unregister and destroy the box. Raise SandboxNotFoundError if unknown."""
        ...

    def reap_idle(self) -> int:
        """Destroy boxes idle longer than idle_ttl that have no exec in flight. Return how many were destroyed."""
        ...

    def health(self) -> HealthResponse:
        """Ping docker, check the configured runtime is installed, report live count and capacity."""
        ...


=
config = ServerConfig()
manager = SandboxManager(config)


async def reaper_loop():
    """Every reap_interval seconds, run manager.reap_idle in a thread. Must survive exceptions."""
    ...


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: manager.startup(), start reaper_loop as a background task.
    Shutdown: cancel the reaper, manager.shutdown()."""
    ...
    yield
    ...


app = FastAPI(title="Sandbox Server", lifespan=lifespan)


# Plain `def` endpoints: FastAPI runs them in a thread pool, so a long exec doesn't block other requests.
# That pool defaults to 40 threads, which caps concurrent execs, raise it in lifespan if needed.


@app.post("/sandboxes", response_model=CreateResponse)
def create_sandbox(req: CreateRequest) -> CreateResponse:
    """manager.create -> CreateResponse. SandboxCapacityError -> HTTPException(503)."""
    ...


@app.post("/sandboxes/{sandbox_id}/exec", response_model=ExecResponse)
def exec_in_sandbox(sandbox_id: str, req: ExecRequest) -> ExecResponse:
    """manager.exec -> ExecResponse. SandboxNotFoundError -> HTTPException(404).
    A dead box, timeout or failing command is a normal 200 response, not an error."""
    ...


@app.delete("/sandboxes/{sandbox_id}")
def destroy_sandbox(sandbox_id: str) -> dict:
    """manager.destroy -> {"ok": True}. SandboxNotFoundError -> HTTPException(404)."""
    ...


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """manager.health()."""
    ...


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port)
