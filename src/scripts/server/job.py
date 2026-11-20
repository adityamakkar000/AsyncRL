import logging
import os
import signal
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import IO, Any

from src.scripts.launch.tpu import Runtime, TPUType, Zone
from src.scripts.server.gcloud import error_tail, tpu_create_queued, tpu_delete_queued, tpu_describe, tpu_node
from src.scripts.server.mesh_config import DEFAULT_USER, IDENTITY_FILE, delete_mesh_config, update_mesh_config

logger = logging.getLogger(__name__)

POLL_SECONDS = 15
MAX_EVENTS = 200
EXIT_GRACE_SECONDS = 60
SETUP_TIMEOUT_SECONDS = 900
RETRY_REFUND_SECONDS = 600


def process_start_time(pid: int) -> str | None:
    res = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True, check=False)
    return res.stdout.strip() or None


class TPUStatus(str, Enum):
    NOT_FOUND = "NOT_FOUND"
    UNKNOWN = "UNKNOWN"
    ACCEPTED = "ACCEPTED"
    CREATING = "CREATING"
    PROVISIONING = "PROVISIONING"
    WAITING_FOR_RESOURCES = "WAITING_FOR_RESOURCES"
    ACTIVE = "ACTIVE"
    SUSPENDING = "SUSPENDING"
    SUSPENDED = "SUSPENDED"
    FAILED = "FAILED"
    DELETING = "DELETING"

    @classmethod
    def parse(cls, value: str) -> "TPUStatus":
        try:
            return cls(value)
        except ValueError:
            return cls.UNKNOWN


PREEMPTED_STATES = {TPUStatus.SUSPENDING, TPUStatus.SUSPENDED, TPUStatus.FAILED}
WAITING_STATES = {TPUStatus.ACCEPTED, TPUStatus.CREATING, TPUStatus.PROVISIONING, TPUStatus.WAITING_FOR_RESOURCES}


class JobState(str, Enum):
    QUEUED = "QUEUED"
    WAITING_FOR_TPU = "WAITING_FOR_TPU"
    SETTING_UP = "SETTING_UP"
    RUNNING = "RUNNING"
    PREEMPTED = "PREEMPTED"
    RELEASING = "RELEASING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_STATES = {JobState.FINISHED, JobState.FAILED, JobState.CANCELLED}


@dataclass
class TPUJob:
    node_id: str
    zone: Zone
    tpu_type: TPUType
    runtime: Runtime
    cmd: str
    cwd: str
    retries: int = 3
    launched_by: str = ""
    state: JobState = JobState.QUEUED
    preemptions: int = 0
    attempts: int = 0
    created_at: float = field(default_factory=time.time)
    state_since: float = field(default_factory=time.time)
    finished_at: float | None = None
    pid: int | None = None
    pid_start: str | None = None
    on_change: Callable[["TPUJob"], None] | None = field(default=None, repr=False)
    tpu_status: str = field(default=TPUStatus.UNKNOWN.value, init=False)
    message: str = field(default="", init=False)
    events: deque = field(default_factory=lambda: deque(maxlen=MAX_EVENTS), init=False, repr=False)
    process: subprocess.Popen | None = field(default=None, init=False, repr=False)
    log_file: IO[Any] | None = field(default=None, init=False, repr=False)
    release_to: JobState | None = field(default=None, init=False)
    exited: tuple[int, float] | None = field(default=None, init=False)
    retry_refundable: bool = field(default=False, init=False)
    attempt_started: float | None = field(default=None, init=False)
    host_ips: list[str] = field(default_factory=list, init=False, repr=False)
    streamers: list[subprocess.Popen] = field(default_factory=list, init=False, repr=False)
    streamer_stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    cancel_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    wake_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    halted: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        is_v5 = self.tpu_type in TPUType.all_v5()
        is_v6 = self.tpu_type in TPUType.all_v6()
        if self.runtime == Runtime.V2_ALPHA_TPUV5 and not is_v5:
            raise ValueError(f"Invalid TPU type {self.tpu_type.value} for runtime {self.runtime.value}")
        if self.runtime == Runtime.V2_ALPHA_TPUV6E and not is_v6:
            raise ValueError(f"Invalid TPU type {self.tpu_type.value} for runtime {self.runtime.value}")
        if is_v5 and self.zone in {Zone.US_EAST1_D, Zone.EUROPE_WEST4_A}:
            raise ValueError(f"TPU type {self.tpu_type.value} is not available in zone {self.zone.value}")
        if is_v6 and self.zone in {Zone.US_EAST5_A, Zone.US_CENTRAL1_A}:
            raise ValueError(f"TPU type {self.tpu_type.value} is not available in zone {self.zone.value}")

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def launch_dir(self) -> str:
        return f"{os.path.expanduser('~')}/{self.cwd}"

    @property
    def command(self) -> str:
        return f'mesh run {self.node_id} "{self.cmd}"'

    @property
    def log_dir(self) -> str:
        return f"{self.launch_dir}/logs/{self.node_id}"

    @property
    def log_path(self) -> str:
        return f"{self.log_dir}/log.txt"

    @property
    def ranks_dir(self) -> str:
        return f"{self.log_dir}/ranks"

    def to_record(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "zone": self.zone.value,
            "tpu_type": self.tpu_type.value,
            "runtime": self.runtime.value,
            "cmd": self.cmd,
            "cwd": self.cwd,
            "retries": self.retries,
            "launched_by": self.launched_by,
            "state": self.state.value,
            "preemptions": self.preemptions,
            "attempts": self.attempts,
            "created_at": self.created_at,
            "state_since": self.state_since,
            "finished_at": self.finished_at,
            "pid": self.pid,
            "pid_start": self.pid_start,
            "release_to": self.release_to.value if self.release_to else None,
            "retry_refundable": self.retry_refundable,
            "events": list(self.events),
        }

    @classmethod
    def from_record(cls, r: dict[str, Any], on_change=None) -> "TPUJob":
        job = cls(
            node_id=r["node_id"],
            zone=Zone(r["zone"]),
            tpu_type=TPUType(r["tpu_type"]),
            runtime=Runtime(r["runtime"]),
            cmd=r["cmd"],
            cwd=r["cwd"],
            retries=r["retries"],
            launched_by=r.get("launched_by", ""),
            state=JobState(r["state"]),
            preemptions=r.get("preemptions", 0),
            attempts=r.get("attempts", 0),
            created_at=r.get("created_at", time.time()),
            state_since=r.get("state_since", time.time()),
            finished_at=r.get("finished_at"),
            pid=r.get("pid"),
            pid_start=r.get("pid_start"),
            on_change=on_change,
        )
        job.events.extend(r.get("events", []))
        job.release_to = JobState(r["release_to"]) if r.get("release_to") else None
        job.retry_refundable = r.get("retry_refundable", False)
        return job

    def event(self, msg: str):
        self.message = msg
        now = time.time()
        last = self.events[-1] if self.events else None
        if last is not None and last[1] == self.state.value and last[2] == msg:
            if len(last) < 5:
                last.extend([1, last[0]])
            last[3] += 1
            last[4] = now
            return
        self.events.append([now, self.state.value, msg, 1, now])
        logger.info("[%s] %s: %s", self.node_id, self.state.value, msg)
        if self.on_change:
            self.on_change(self)

    def set_state(self, state: JobState, msg: str):
        if state != self.state:
            self.state = state
            self.state_since = time.time()
            if state in TERMINAL_STATES:
                self.finished_at = time.time()
            self.event(msg)
        elif msg != self.message:
            self.event(msg)

    def start(self):
        if not self.is_terminal:
            self.thread = threading.Thread(target=self.run, name=f"job-{self.node_id}", daemon=True)
            self.thread.start()

    def cancel(self):
        self.cancel_event.set()
        self.wake_event.set()

    def halt(self):
        self.halted.set()
        self.wake_event.set()
        self.stop_rank_streamers()

    def force_remove(self):
        self.halt()
        if self.thread is not None:
            self.thread.join(timeout=5)
        self.stop_process()
        self.release_to = None
        self.set_state(
            JobState.CANCELLED,
            f"force removed, TPU {self.node_id} may still exist: gcloud compute tpus queued-resources list --zone={self.zone.value}",
        )

    def run(self):
        if self.pid is not None:
            self.kill_stale_process()
        while not self.is_terminal and not self.halted.is_set():
            try:
                self.step()
            except Exception as e:
                logger.exception("[%s] step failed", self.node_id)
                self.event(f"internal error: {e!r}, retrying")
            if self.is_terminal or self.halted.is_set():
                break
            self.wake_event.wait(POLL_SECONDS)
            self.wake_event.clear()

    def step(self):
        status = TPUStatus.parse(tpu_describe(self.node_id, self.zone.value))
        self.tpu_status = status.value

        if self.cancel_event.is_set() and self.release_to != JobState.CANCELLED:
            self.begin_release(JobState.CANCELLED, "cancel requested, releasing TPU")
            return
        if self.release_to is not None:
            self.release(status)
            return

        if status in PREEMPTED_STATES:
            self.preempted(f"TPU {status.value}: preempted, requeueing")
            return

        if self.process is not None:
            code = self.process.poll()
            if code is None:
                if self.retry_refundable and time.time() - (self.attempt_started or 0) > RETRY_REFUND_SECONDS:
                    self.retry_refundable = False
                if self.state != JobState.RUNNING:
                    self.set_state(JobState.RUNNING, f"running (attempt {self.attempts})")
                return
            self.stop_process()
            if code == 0:
                self.begin_release(JobState.FINISHED, "job finished, releasing TPU")
                return
            self.exited = (code, time.time())
            self.event(f"job exited {code}, waiting {EXIT_GRACE_SECONDS}s to rule out preemption before retrying")
            return

        if self.exited is not None:
            code, at = self.exited
            if time.time() - at < EXIT_GRACE_SECONDS:
                return
            self.exited = None
            node_state, _ = tpu_node(self.node_id, self.zone.value)
            if node_state not in {"READY", "UNKNOWN"}:
                self.preempted(f"job exited {code} with TPU node {node_state}: preempted, requeueing")
                return
            if self.retries <= 0:
                self.begin_release(JobState.FAILED, f"job exited {code}, no retries left, releasing TPU")
                return
            self.retries -= 1
            self.retry_refundable = True
            self.event(f"job exited {code}, retrying on same TPU ({self.retries} retries left)")

        if status == TPUStatus.NOT_FOUND:
            res = tpu_create_queued(self.node_id, self.tpu_type.value, self.runtime.value, self.zone.value)
            if res.returncode == 0:
                self.set_state(JobState.WAITING_FOR_TPU, f"requested spot {self.tpu_type.value} in {self.zone.value}")
            else:
                self.set_state(JobState.QUEUED, f"TPU request failed, retrying: {error_tail(res)}")
        elif status in WAITING_STATES:
            self.set_state(JobState.WAITING_FOR_TPU, f"TPU {status.value}")
        elif status == TPUStatus.DELETING:
            self.set_state(JobState.QUEUED, "waiting for previous TPU to finish deleting")
        elif status == TPUStatus.ACTIVE:
            self.launch()
        else:
            self.event("could not read TPU status, retrying")

    def launch(self):
        if self.state != JobState.SETTING_UP:
            self.set_state(JobState.SETTING_UP, "TPU active, running mesh setup")
        node_state, ips = tpu_node(self.node_id, self.zone.value)
        if node_state != "READY" or not ips:
            self.event(f"TPU node {node_state} with {len(ips)} hosts, waiting")
            return
        self.host_ips = ips
        update_mesh_config(self.node_id, ips)
        try:
            res = subprocess.run(
                ["mesh", "setup", self.node_id],
                capture_output=True,
                text=True,
                cwd=self.launch_dir,
                check=False,
                timeout=SETUP_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            self.event(f"mesh setup timed out after {SETUP_TIMEOUT_SECONDS}s, retrying")
            return
        if res.returncode != 0:
            self.event(f"mesh setup failed, retrying: {error_tail(res)}")
            return
        if self.cancel_event.is_set() or self.halted.is_set():
            return
        os.makedirs(self.ranks_dir, exist_ok=True)
        self.log_file = open(self.log_path, "a", buffering=1)
        self.process = subprocess.Popen(
            self.command,
            shell=True,
            stdout=self.log_file,
            stderr=self.log_file,
            text=True,
            cwd=self.launch_dir,
            start_new_session=True,
        )
        self.pid = self.process.pid
        self.pid_start = process_start_time(self.pid)
        self.attempts += 1
        self.attempt_started = time.time()
        proc = self.process
        threading.Thread(target=lambda: (proc.wait(), self.wake_event.set()), daemon=True).start()
        self.start_rank_streamers()
        self.set_state(JobState.RUNNING, f"running on {len(ips)} hosts (attempt {self.attempts})")

    def preempted(self, msg: str):
        self.preemptions += 1
        if self.retry_refundable:
            self.retries += 1
            self.retry_refundable = False
            msg += f", retry refunded ({self.retries} left)"
        self.begin_release(JobState.QUEUED, msg)

    def begin_release(self, target: JobState, msg: str):
        self.release_to = target
        self.exited = None
        self.set_state(JobState.PREEMPTED if target == JobState.QUEUED else JobState.RELEASING, msg)
        self.release(TPUStatus.parse(self.tpu_status))

    def release(self, status: TPUStatus):
        self.stop_process()
        if status not in {TPUStatus.NOT_FOUND, TPUStatus.DELETING}:
            res = tpu_delete_queued(self.node_id, self.zone.value)
            if res.returncode != 0:
                self.event(f"TPU delete failed, retrying (server_delete --force drops the job): {error_tail(res)}")
                return
        try:
            delete_mesh_config(self.node_id)
        except Exception:
            logger.exception("[%s] could not remove mesh config entry", self.node_id)
        self.tpu_status = TPUStatus.NOT_FOUND.value
        target, self.release_to = self.release_to, None
        if target == JobState.QUEUED:
            self.set_state(JobState.QUEUED, f"TPU released after preemption #{self.preemptions}, requesting a new one")
            self.wake_event.set()
        elif target == JobState.FINISHED:
            self.set_state(JobState.FINISHED, f"finished after {self.attempts} attempts, TPU released")
        elif target == JobState.FAILED:
            self.set_state(JobState.FAILED, "failed, TPU released")
        else:
            self.set_state(JobState.CANCELLED, "cancelled, TPU released")

    def stop_process(self):
        self.stop_rank_streamers()
        if self.process is not None:
            self.kill_group(self.process.pid)
            self.process = None
        self.pid = None
        self.pid_start = None
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def kill_group(self, pid: int):
        for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 0)):
            try:
                os.killpg(pid, sig)
            except (ProcessLookupError, PermissionError):
                return
            deadline = time.time() + wait
            while time.time() < deadline:
                try:
                    os.killpg(pid, 0)
                except ProcessLookupError:
                    return
                time.sleep(0.5)

    def kill_stale_process(self):
        assert self.pid is not None
        pid, started = self.pid, self.pid_start
        self.pid, self.pid_start = None, None
        if started is not None and process_start_time(pid) == started:
            self.event(f"server restarted, stopping stale run process {pid}")
            self.kill_group(pid)
        else:
            self.event(f"server restarted, previous run process {pid} is gone")

    def start_rank_streamers(self):
        self.stop_rank_streamers()
        stop = threading.Event()
        self.streamer_stop = stop
        for i, ip in enumerate(self.host_ips):
            threading.Thread(target=self.stream_rank, args=(i, ip, stop), daemon=True).start()

    def stream_rank(self, index: int, ip: str, stop: threading.Event):
        path = f"{self.ranks_dir}/w{index}.txt"
        first = True
        while not stop.is_set():
            n = "+1" if first else "0"
            first = False
            with open(path, "a", buffering=1) as f:
                proc = subprocess.Popen(
                    [
                        "ssh",
                        "-o",
                        "BatchMode=yes",
                        "-o",
                        "StrictHostKeyChecking=no",
                        "-o",
                        "UserKnownHostsFile=/dev/null",
                        "-o",
                        "LogLevel=ERROR",
                        "-o",
                        "ConnectTimeout=15",
                        "-o",
                        "ServerAliveInterval=30",
                        "-i",
                        IDENTITY_FILE,
                        f"{DEFAULT_USER}@{ip}",
                        f"tail -n {n} -F ~/job/output.log",
                    ],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                )
                self.streamers.append(proc)
                while proc.poll() is None and not stop.is_set():
                    time.sleep(1)
                if proc.poll() is None:
                    proc.terminate()
            stop.wait(15)

    def stop_rank_streamers(self):
        self.streamer_stop.set()
        for proc in self.streamers:
            if proc.poll() is None:
                proc.terminate()
        self.streamers.clear()
