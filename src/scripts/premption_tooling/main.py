import subprocess
import time
from dataclasses import dataclass

from config_utils import update_mesh_config
from gcp_utils import tpu_create_queued, tpu_delete_queued, tpu_describe, tpu_get_ips
from rich.console import Console
from rich.live import Live
from rich.table import Table


@dataclass
class TPUJob:
    node_id: str
    zone: str
    tpu_type: str
    runtime: str
    cmd: str


console = Console()


def generate_table(active_jobs, job_states, job_processes) -> Table:
    """Creates a Rich Table UI of current job statuses."""
    table = Table(title="TPU Cluster Dashboard", title_style="bold magenta")

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

                            tpu_delete_queued(node_id, zone)
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
                    if is_running_locally:
                        proc.terminate()

                    job_processes[node_id] = None
                    if state != "NOT_FOUND":
                        tpu_delete_queued(node_id, zone)
                        time.sleep(5)
                    tpu_create_queued(node_id, job.tpu_type, job.runtime, zone, spot=True)

            time.sleep(60)
            live.update(generate_table(active_jobs, job_states, job_processes))


if __name__ == "__main__":
    MY_RUNS = [
        TPUJob(
            node_id="node237",
            zone="us-central1-a",
            tpu_type="v5p-8",
            runtime="v2-alpha-tpuv5",
            cmd="python -m src.train --config-name debug experiment_name=stop_clip2",
        )
    ]
    run_session(MY_RUNS)
