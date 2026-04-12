import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from rich.console import Console
from rich.live import Live
from rich.table import Table

from .config_utils import update_mesh_config
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
    process: subprocess.Popen | None = None

    def __post_init__(self):
        is_v5 = self.tpu_type in {TPUType.V5P_8, TPUType.V5P_32, TPUType.V5P_64, TPUType.V5P_128}
        is_v6 = self.tpu_type in {TPUType.V6E_8, TPUType.V6E_32, TPUType.V6E_64, TPUType.V6E_128}
        if self.runtime == Runtime.V2_ALPHA_TPUV5 and not is_v5:
            raise ValueError(f"Invalid TPU type {self.tpu_type} for runtime {self.runtime}")
        if self.runtime == Runtime.V2_ALPHA_TPUV6E and not is_v6:
            raise ValueError(f"Invalid TPU type {self.tpu_type} for runtime {self.runtime}")
        if is_v5 and self.zone in {Zone.US_EAST1_D, Zone.EUROPE_WEST4_A}:
            raise ValueError(f"TPU type {self.tpu_type} is not available in zone {self.zone}")
        if is_v6 and self.zone not in {Zone.US_EAST5_A, Zone.US_CENTRAL1_A}:
            raise ValueError(f"TPU type {self.tpu_type} is not available in zone {self.zone}")
        if self.process is not None:
            raise ValueError("Process should be initialized to None")

    @property
    def tpu_status(self) -> str:
        return tpu_describe(self.node_id, self.zone.value)

    @property
    def is_tpu_active(self) -> bool:
        return self.tpu_status == TPUStatus.ACTIVE.value

    @property
    def is_tpu_prempted(self) -> bool:
        return self.setup_tpu() != 0

    @property
    def job_status(self) -> str:
        if self.process is None:
            return "PENDING"

        if (ec := self.process.poll()) is None:
            return "RUNNING"

        if self.is_tpu_prempted:
            return "PREEMPTED"

        return "FINISHED" if ec == 0 else "ERROR"

    @property
    def is_job_finished(self) -> bool:
        return self.job_status in ["FINISHED", "ERROR"]

    def setup_tpu(self) -> int:
        if not self.is_tpu_active:
            return 1
        ips = tpu_get_ips(self.node_id, self.zone.value)
        update_mesh_config(self.node_id, ips)
        cmd = subprocess.run(["mesh", "setup", self.node_id], capture_output=True)
        return cmd.returncode

    def launch_job(self):
        if not self.is_tpu_active:
            self.allocate_tpu()
            return

        mesh_setup_code = self.setup_tpu()
        if mesh_setup_code != 0:
            return

        full_cmd = f'mesh run {self.node_id} "{self.cmd}"'
        log_file = f"log_{self.node_id}.txt"
        f = open(log_file, "a", buffering=1)

        self.process = subprocess.Popen(full_cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, text=True)

    def allocate_tpu(self):
        if self.tpu_status == TPUStatus.NOT_FOUND.value:
            tpu_create_queued(self.node_id, self.tpu_type.value, self.runtime.value, self.zone.value, spot=True)

    def delete_tpu(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()

        if self.tpu_status in TPUStatus.allocated_states():
            tpu_delete_queued(self.node_id, self.zone.value)

    def check_and_handle_premption(self):
        if (js := self.job_status) == "PREEMPTED" or self.tpu_status in TPUStatus.failed_states():
            self.delete_tpu()
            self.process = None
        elif js == "PENDING":
            self.launch_job()
        elif js == "ERROR":
            self.delete_tpu()


def generate_table(jobs: list[TPUJob]) -> Table:
    current_time: str = datetime.now().strftime("%H:%M:%S")
    title = f"TPU Cluster Dashboard [dim](Last Updated: {current_time})[/dim]"

    table = Table(title=title, title_style="bold magenta")

    table.add_column("Node ID", style="cyan", no_wrap=True)
    table.add_column("TPU Status", style="yellow")
    table.add_column("Job Status", style="green")
    table.add_column("Job State", style="blue")
    table.add_column("Command", style="dim", overflow="ellipsis")
    table.add_column("Log Tail Command", style="blue")

    for job in jobs:
        log_cmd = f"tail -f log_{job.node_id}.txt"
        table.add_row(
            job.node_id,
            job.tpu_status,
            job.job_status,
            job.cmd[:40] + "..." if len(job.cmd) > 40 else job.cmd,
            log_cmd,
        )
    return table


def run_session(jobs: list[TPUJob]):
    try:
        with Live(generate_table(jobs), refresh_per_second=1) as live:
            while any(not j.is_job_finished for j in jobs):
                for j in jobs:
                    j.check_and_handle_premption()
                live.update(generate_table(jobs))
                time.sleep(UPDATE_TIME)
    finally:
        console.print("[bold red]Cleaning up...[/bold red]")
        for j in jobs:
            j.delete_tpu()
        console.print("[bold green]Cleanup complete.[/bold green]")
