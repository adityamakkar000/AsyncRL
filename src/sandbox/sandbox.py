import time
import uuid
from dataclasses import dataclass
from typing import Self

import docker
from docker.errors import APIError, NotFound
from docker.models.containers import Container

LABEL = "annealedrl.sandbox"
TIMEOUT_EXIT_CODE = 124

EXEC_WRAPPER = 'ulimit -v "$1" && exec timeout -k 1 "$2" sh -c "$3"'


@dataclass
class SandboxConfig:
    image: str = "python:3.12-slim"
    runtime: str | None = "runsc"  # gVisor; None falls back to docker's default runtime (e.g. local dev on macOS)
    cpus: float = 1.0
    memory_mb: int = 1024  # hard cap for the whole box; under gVisor exceeding it kills the box
    process_memory_mb: int = 768  # per-process cap (ulimit -v) so a greedy program gets MemoryError instead
    network: bool = False
    user: str = "65534:65534"  # nobody
    workdir: str = "/workspace"
    workdir_mb: int = 256  # tmpfs, counts towards the memory cap
    max_output_bytes: int = 64 * 1024  # per stream; the rest is read and dropped


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool
    duration: float
    sandbox_died: bool = False  
    stdout_truncated: bool = False
    stderr_truncated: bool = False


def make_client(max_pool_size: int = 64) -> docker.DockerClient:
    """One client can be shared by many sandboxes and threads; the pool bounds concurrent docker API calls."""
    return docker.from_env(max_pool_size=max_pool_size)


class Sandbox:
    """A single, non-root locked-down container. Only `workdir` and /tmp are writable."""

    def __init__(self, config: SandboxConfig, client: docker.DockerClient | None = None):
        self.config = config
        self.client = client or make_client()
        self.container: Container | None = None

    def start(self) -> Self:
        cfg = self.config
        if cfg.process_memory_mb + cfg.workdir_mb > cfg.memory_mb:
            raise ValueError(
                "process_memory_mb + workdir_mb must fit inside memory_mb, otherwise the box can be killed"
            )

        self.container = self.client.containers.run(
            cfg.image,
            command=["sleep", "infinity"],
            name=f"sandbox-{uuid.uuid4().hex[:12]}",
            labels={LABEL: "1"},
            detach=True,
            init=True,
            runtime=cfg.runtime,
            user=cfg.user,
            working_dir=cfg.workdir,
            environment={"HOME": cfg.workdir},
            read_only=True,
            tmpfs={
                cfg.workdir: f"rw,size={cfg.workdir_mb}m,mode=1777",
                "/tmp": "rw,size=64m,mode=1777",
            },
            network_mode="bridge" if cfg.network else "none",
            nano_cpus=int(cfg.cpus * 1e9),
            mem_limit=f"{cfg.memory_mb}m",
            security_opt=["no-new-privileges"],
        )
        return self

    def exec(self, command: str, timeout: float = 30.0) -> ExecResult:
        if self.container is None:
            raise RuntimeError("Sandbox is not started")

        api = self.client.api
        cap = self.config.max_output_bytes
        argv = ["sh", "-c", EXEC_WRAPPER, "sh", str(self.config.process_memory_mb * 1024), str(timeout), command]
        stdout, stderr = bytearray(), bytearray()

        start = time.perf_counter()
        try:
            exec_id = api.exec_create(self.container.id, argv)["Id"]
            # Stream and stop keeping output past the cap, so a command printing gigabytes never sits in memory.
            # Each buffer can overshoot the cap by one chunk, which is how truncation is detected below.
            for out, err in api.exec_start(exec_id, stream=True, demux=True):
                if out and len(stdout) <= cap:
                    stdout += out
                if err and len(stderr) <= cap:
                    stderr += err
            exit_code = api.exec_inspect(exec_id)["ExitCode"]
        except (APIError, NotFound):
            if self.is_alive():
                raise  # docker itself failed, not the box
            exit_code = None
        duration = time.perf_counter() - start

        # A dead box shows up as an odd exit code (gVisor reports 128, a killed box 137) rather than an
        # exception, so probe the box whenever the exit code isn't a normal one.
        sandbox_died = exit_code is None or (exit_code >= 125 and not self.is_alive())

        return ExecResult(
            stdout=stdout[:cap].decode(errors="replace"),
            stderr=stderr[:cap].decode(errors="replace"),
            exit_code=exit_code,
            # 137 = `timeout` had to force-kill a command that ignored the first signal
            timed_out=exit_code == TIMEOUT_EXIT_CODE or (exit_code == 137 and not sandbox_died and duration >= timeout),
            duration=duration,
            sandbox_died=sandbox_died,
            stdout_truncated=len(stdout) > cap,
            stderr_truncated=len(stderr) > cap,
        )

    def is_alive(self) -> bool:
        """Run a no-op in the box. Docker's own status lags behind for a moment after a box dies,
        so asking the box itself is the reliable check."""
        if self.container is None:
            return False
        try:
            exit_code, _ = self.container.exec_run(["true"])
        except (APIError, NotFound):
            return False
        return exit_code == 0

    def destroy(self):
        if self.container is not None:
            try:
                self.container.remove(force=True)
            except NotFound:
                pass
            self.container = None

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *_):
        self.destroy()


def cleanup_all(client: docker.DockerClient | None = None) -> int:
    """Remove every sandbox container left on this host, e.g. after a crash."""
    client = client or make_client()
    containers = client.containers.list(all=True, filters={"label": LABEL})
    for c in containers:
        c.remove(force=True)
    return len(containers)
