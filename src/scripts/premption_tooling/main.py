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


@dataclass
class TPUJob:
    node_id: str
    zone: Zone
    tpu_type: TPUType
    runtime: Runtime
    cmd: str

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


console = Console()


def generate_table(active_jobs, job_states, job_processes) -> Table:
    current_time: str = datetime.now().strftime("%H:%M:%S")
    title = f"TPU Cluster Dashboard [dim](Last Updated: {current_time})[/dim]"

    table = Table(title=title, title_style="bold magenta")

    table.add_column("Node ID", style="cyan", no_wrap=True)
    table.add_column("TPU State", style="yellow")
    table.add_column("Process", style="green")
    table.add_column("Command", style="dim", overflow="ellipsis")
    table.add_column("Log Tail Command", style="blue")

    for job in active_jobs:
        node_id = job.node_id
        tpu_state = job_states.get(node_id, "Unknown")

        proc = job_processes.get(node_id)
        if proc is None:
            process_status = "[bold red]Not Started[/bold red]"
        elif proc.poll() is None:
            process_status = "[bold green]Running[/bold green]"
        else:
            process_status = f"[bold white]Exited ({proc.returncode})[/bold white]"

        log_cmd = f"tail -f log_{node_id}.txt"

        table.add_row(
            node_id, tpu_state, process_status, job.cmd[:40] + "..." if len(job.cmd) > 40 else job.cmd, log_cmd
        )
    return table


def run_session(initial_jobs: list[TPUJob]):
    active_jobs = list(initial_jobs)
    job_processes = {job.node_id: None for job in active_jobs}
    job_states = {job.node_id: "INIT" for job in active_jobs}

    with Live(generate_table(active_jobs, job_states, job_processes), refresh_per_second=1) as live:
        while active_jobs:
            for job in active_jobs[:]:
                node_id = job.node_id
                zone = job.zone

                state = tpu_describe(node_id, zone)
                job_states[node_id] = state

                live.update(generate_table(active_jobs, job_states, job_processes))

                proc = job_processes.get(node_id)
                is_running_locally = proc is not None and proc.poll() is None

                if state == "ACTIVE":
                    if not is_running_locally:
                        if proc is not None:
                            exit_code = proc.poll()
                            console.print(f"[bold red]CRITICAL:[/bold red] {node_id} exited with code {exit_code}")

                            active_jobs.remove(job)
                            continue

                        ips = tpu_get_ips(node_id, zone)
                        update_mesh_config(node_id, ips)
                        subprocess.run(["mesh", "setup", node_id], capture_output=True)

                        full_cmd = f'mesh run {node_id} "{job.cmd}"'
                        log_file = f"log_{node_id}.txt"
                        f = open(log_file, "a", buffering=1)

                        job_processes[node_id] = subprocess.Popen(
                            full_cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, text=True
                        )

                elif state in ["FAILED", "SUSPENDED", "NOT_FOUND"]:
                    if is_running_locally and proc is not None:
                        proc.terminate()

                    job_processes[node_id] = None
                    if state != "NOT_FOUND":
                        tpu_delete_queued(node_id, zone)
                        time.sleep(5)
                    tpu_create_queued(node_id, job.tpu_type, job.runtime, zone, spot=True)

            time.sleep(UPDATE_TIME)
            live.update(generate_table(active_jobs, job_states, job_processes))
