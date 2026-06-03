import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rich.console import Console

from .config_utils import delete_mesh_config, update_mesh_config
from .gcp_utils import tpu_create_queued, tpu_delete_queued, tpu_describe, tpu_get_ips

console = Console()


class Zone(str, Enum):
    US_CENTRAL1_A = "us-central1-a"
    US_EAST5_A = "us-east5-a"
    US_EAST1_D = "us-east1-d"
    EUROPE_WEST4_A = "europe-west4-a"


class TPUType(str, Enum):
    V5P_8 = "v5p-8"
    V5P_16 = "v5p-16"
    V5P_32 = "v5p-32"
    V5P_64 = "v5p-64"
    V5P_128 = "v5p-128"
    V6E_8 = "v6e-8"
    V6E_32 = "v6e-32"
    V6E_64 = "v6e-64"
    V6E_128 = "v6e-128"

    @staticmethod
    def all_v5() -> list["TPUType"]:
        return [TPUType.V5P_8, TPUType.V5P_32, TPUType.V5P_64, TPUType.V5P_128]

    @staticmethod
    def all_v6() -> list["TPUType"]:
        return [TPUType.V6E_8, TPUType.V6E_32, TPUType.V6E_64, TPUType.V6E_128]


class Runtime(str, Enum):
    V2_ALPHA_TPUV5 = "v2-alpha-tpuv5"
    V2_ALPHA_TPUV6E = "v2-alpha-tpuv6e"


class TPUStatus(str, Enum):
    ACTIVE = "ACTIVE"
    PREEMPTED = "PREEMPTED"
    SUSPENDED = "SUSPENDED"
    FAILED = "FAILED"
    NOT_FOUND = "NOT_FOUND"

    @staticmethod
    def failed_states() -> list[str]:
        return [TPUStatus.FAILED.value, TPUStatus.PREEMPTED.value, TPUStatus.SUSPENDED.value]

    @staticmethod
    def allocated_states() -> list[str]:
        return [TPUStatus.ACTIVE.value, TPUStatus.PREEMPTED.value, TPUStatus.SUSPENDED.value, TPUStatus.FAILED.value]


@dataclass
class TPUJob:
    node_id: str
    zone: Zone
    tpu_type: TPUType
    runtime: Runtime
    cmd: str
    cwd: str
    process: subprocess.Popen | None = None
    retries: int = 3
    launched_by: str = ""
    keep_logs: bool = True

    _log_file: Any = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _tpu_cached_status: str = field(default="NOT_FOUND", init=False)

    def __post_init__(self):
        is_v5 = self.tpu_type in TPUType.all_v5()
        is_v6 = self.tpu_type in TPUType.all_v6()
        if self.runtime == Runtime.V2_ALPHA_TPUV5 and not is_v5:
            raise ValueError(f"Invalid TPU architecture combination: {self.tpu_type} with {self.runtime}")
        if self.runtime == Runtime.V2_ALPHA_TPUV6E and not is_v6:
            raise ValueError(f"Invalid TPU architecture combination: {self.tpu_type} with {self.runtime}")
        if is_v5 and self.zone in {Zone.US_EAST1_D, Zone.EUROPE_WEST4_A}:
            raise ValueError(f"TPU Type {self.tpu_type} is physically absent from zone {self.zone.value}")
        if is_v6 and self.zone in {Zone.US_EAST5_A, Zone.US_CENTRAL1_A}:
            raise ValueError(f"TPU Type {self.tpu_type} is physically absent from zone {self.zone.value}")

    @property
    def tpu_status(self) -> str:
        with self._lock:
            return self._tpu_cached_status

    @property
    def job_status(self) -> str:
        with self._lock:
            if self.process is None:
                return "PENDING"
            ec = self.process.poll()
            if ec is None:
                return "RUNNING"
            return "FINISHED" if ec == 0 else "ERROR"

    @property
    def is_job_finished(self) -> bool:
        return self.job_status == "FINISHED"

    @property
    def is_job_finished_without_error(self) -> bool:
        return self.job_status == "FINISHED"

    @property
    def home_dir(self) -> str:
        return os.path.expanduser("~")

    @property
    def launch_dir(self) -> str:
        return os.path.join(self.home_dir, self.cwd) if self.cwd else self.home_dir

    def update_cluster_telemetry(self) -> None:
        """Executed inside background thread pool to refresh cached cloud states safely."""
        try:
            status = tpu_describe(self.node_id, self.zone.value)
            with self._lock:
                self._tpu_cached_status = status
        except Exception:
            console.print(f"[red]Failed querying cloud descriptor metadata for node: {self.node_id}[/red]")

    def setup_tpu(self) -> int:
        ips = tpu_get_ips(self.node_id, self.zone.value)
        update_mesh_config(self.node_id, ips)
        result = subprocess.run(["mesh", "setup", self.node_id], capture_output=True, cwd=self.launch_dir)
        return result.returncode

    def launch_job(self) -> None:
        with self._lock:
            if self._tpu_cached_status != TPUStatus.ACTIVE.value:
                if self._tpu_cached_status == TPUStatus.NOT_FOUND.value:
                    self.allocate_tpu()
                return

            if self.process is not None and self.process.poll() is None:
                return  # Guard clause against duplicate tracking launches

            if self.setup_tpu() != 0:
                console.print(
                    f"[yellow]Mesh config verification handshake failed for {self.node_id}. Retrying...[/yellow]"
                )
                return

            full_cmd = f'mesh run {self.node_id} "{self.cmd}"'
            log_dir = os.path.join(self.home_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"{self.node_id}.txt")

            self._log_file = open(log_path, "a", buffering=1, encoding="utf-8")
            self.process = subprocess.Popen(
                full_cmd,
                shell=True,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.launch_dir,
            )

    def allocate_tpu(self) -> None:
        tpu_create_queued(
            self.node_id,
            self.tpu_type.value,
            self.runtime.value,
            self.zone.value,
            spot=True,
        )

    def delete_tpu(self) -> None:
        with self._lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()

            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None

            if self._tpu_cached_status in TPUStatus.allocated_states():
                tpu_delete_queued(self.node_id, self.zone.value)
                delete_mesh_config(self.node_id)
                self._tpu_cached_status = "NOT_FOUND"

    def check_and_handle_preemption(self) -> None:
        self.update_cluster_telemetry()
        js = self.job_status
        tpu_stat = self.tpu_status

        if tpu_stat in TPUStatus.failed_states():
            console.print(
                f"[yellow]{self.node_id} encountered infrastructure interruption ({tpu_stat}). Triggering rollover recovery...[/yellow]"
            )
            self.delete_tpu()
            with self._lock:
                self.process = None

        elif js == "PENDING":
            self.launch_job()

        elif js == "FINISHED" and tpu_stat in TPUStatus.allocated_states():
            console.print(f"[green]Job completed sequence on {self.node_id}. Releasing cloud allocations.[/green]")
            self.delete_tpu()


def run_session(jobs: list[TPUJob], executor: ThreadPoolExecutor) -> None:
    futures = [executor.submit(j.check_and_handle_preemption) for j in jobs]
    for fut in futures:
        try:
            fut.result()
        except Exception:
            console.print_exception()
