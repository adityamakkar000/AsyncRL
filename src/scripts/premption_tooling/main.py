import os
import subprocess
import threading
from dataclasses import dataclass, field
from enum import Enum

from rich.console import Console

from .config_utils import delete_mesh_config, update_mesh_config
from .gcp_utils import tpu_create_queued, tpu_delete_queued, tpu_describe, tpu_get_ips

UPDATE_TIME = 10
console = Console()


class Zone(str, Enum):
    US_CENTRAL1_A = "us-central1-a"
    US_EAST5_A = "us-east5-a"
    US_EAST1_D = "us-east1-d"
    EUROPE_WEST4_A = "europe-west4-a"


class TPUType(str, Enum):
    V5P_8 = "v5p-8"
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
        return [
            TPUStatus.FAILED.value,
            TPUStatus.PREEMPTED.value,
            TPUStatus.SUSPENDED.value,
        ]

    @staticmethod
    def allocated_states() -> list[str]:
        return [
            TPUStatus.ACTIVE.value,
            TPUStatus.PREEMPTED.value,
            TPUStatus.SUSPENDED.value,
            TPUStatus.FAILED.value,
        ]


@dataclass
class TPUJob:
    node_id: str
    zone: Zone
    tpu_type: TPUType
    runtime: Runtime
    cmd: str
    cwd: str
    process: subprocess.Popen | None = None
    _log_file: object = field(default=None, init=False, repr=False)
    retries: int = 3
    launched_by: str = ""
    cleanup_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    cleanup_done: bool = field(default=False, init=False, repr=False)
    keep_logs: bool = True  # log file will be kept unless keep_logs is False

    def __post_init__(self):
        is_v5 = self.tpu_type in TPUType.all_v5()
        is_v6 = self.tpu_type in TPUType.all_v6()
        if self.runtime == Runtime.V2_ALPHA_TPUV5 and not is_v5:
            raise ValueError(f"Invalid TPU type {self.tpu_type} for runtime {self.runtime}")
        if self.runtime == Runtime.V2_ALPHA_TPUV6E and not is_v6:
            raise ValueError(f"Invalid TPU type {self.tpu_type} for runtime {self.runtime}")
        if is_v5 and self.zone in {Zone.US_EAST1_D, Zone.EUROPE_WEST4_A}:
            raise ValueError(f"TPU type {self.tpu_type} is not available in zone {self.zone}")
        if is_v6 and self.zone in {Zone.US_EAST5_A, Zone.US_CENTRAL1_A}:
            raise ValueError(f"TPU type {self.tpu_type} is not available in zone {self.zone}")
        if self.process is not None:
            raise ValueError("Process should be initialized to None")

    @property
    def tpu_status(self) -> str:
        return tpu_describe(self.node_id, self.zone.value)

    @property
    def job_status(self) -> str:
        if self.process is None:
            return "PENDING"
        if (ec := self.process.poll()) is None:
            return "RUNNING"
        return "FINISHED" if ec == 0 else "ERROR"

    @property
    def is_job_finished(self) -> bool:
        return self.job_status in {"FINISHED", "ERROR"}

    @property
    def is_job_finished_without_error(self) -> bool:
        return self.job_status in {"FINISHED"}

    @property
    def home_dir(self) -> str:
        return os.path.expanduser("~")

    @property
    def launch_dir(self) -> str:
        return f"{self.home_dir}/{self.cwd}"

    @property
    def command(self) -> str:
        return f'mesh run {self.node_id} "{self.cmd}"'

    def setup_tpu(self) -> int:
        ips = tpu_get_ips(self.node_id, self.zone.value)
        update_mesh_config(self.node_id, ips)
        result = subprocess.run(["mesh", "setup", self.node_id], capture_output=True, cwd=self.launch_dir)
        return result.returncode

    def launch_job(self):
        with self.cleanup_lock:
            tpu_status = self.tpu_status

            if tpu_status != TPUStatus.ACTIVE.value:
                if tpu_status == TPUStatus.NOT_FOUND.value:
                    self.allocate_tpu()
                return

            if self.setup_tpu() != 0:
                console.print(f"[yellow]mesh setup failed for {self.node_id}, will retry[/yellow]")
                return

            log_path = f"{self.home_dir}/logs/{self.node_id}.txt"

            self._log_file = open(log_path, "a", buffering=1)
            self.process = subprocess.Popen(
                self.command,
                shell=True,
                stdout=self._log_file,
                stderr=self._log_file,
                text=True,
                cwd=self.launch_dir,
            )

    def allocate_tpu(self):
        tpu_create_queued(
            self.node_id,
            self.tpu_type.value,
            self.runtime.value,
            self.zone.value,
            spot=True,
        )

    def delete_tpu(self):
        with self.cleanup_lock:
            if self.cleanup_done:
                return

            if self.process is not None and self.process.poll() is None:
                self.process.terminate()

            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None

            if self.tpu_status in TPUStatus.allocated_states():
                tpu_delete_queued(self.node_id, self.zone.value)
                delete_mesh_config(self.node_id)

            self.cleanup_done = True

    def check_and_handle_preemption(self):
        js = self.job_status
        tpu_status = self.tpu_status

        if tpu_status in TPUStatus.failed_states():
            console.print(f"[yellow]{self.node_id} preempted/failed (TPU: {tpu_status}), re-queuing...[/yellow]")
            self.delete_tpu()
            with self.cleanup_lock:
                self.process = None

        elif js == "ERROR":
            console.print(f"[red]{self.node_id} job errored, releasing TPU[/red]")
            if self.retries > 0:
                console.print(f"[yellow]Retrying {self.node_id} (retries left: {self.retries})[/yellow]")
                self.retries -= 1
                with self.cleanup_lock:
                    self.process = None
            else:
                self.delete_tpu()

        elif js == "PENDING":
            with self.cleanup_lock:
                self.cleanup_done = False
            self.launch_job()

        elif js == "FINISHED" and tpu_status in TPUStatus.allocated_states():
            console.print(f"[green]{self.node_id} job finished, releasing TPU[/green]")
            self.delete_tpu()


def run_session(jobs: list[TPUJob]):
    threads = [threading.Thread(target=j.check_and_handle_preemption) for j in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
