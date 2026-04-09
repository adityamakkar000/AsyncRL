import subprocess
import time
from dataclasses import dataclass

from config_utils import update_mesh_config
from gcp_utils import tpu_create_queued, tpu_delete_queued, tpu_describe, tpu_get_ips


@dataclass
class TPUJob:
    node_id: str
    zone: str
    tpu_type: str
    runtime: str
    cmd: str


def run_session(initial_jobs: list[TPUJob]):
    active_jobs = list(initial_jobs)
    job_processes = {job.node_id: None for job in active_jobs}

    print(f"Monitoring {len(active_jobs)} active TPU runs...")

    while active_jobs:
        for job in active_jobs[:]:
            node_id = job.node_id
            zone = job.zone
            state = tpu_describe(node_id, zone)

            proc = job_processes.get(node_id)
            is_running_locally = proc is not None and proc.poll() is None

            print(f"[{node_id}]: Hardware={state}, Process_Alive={is_running_locally}")

            if state == "ACTIVE":
                if not is_running_locally:
                    if proc is not None:
                        exit_code = proc.poll()
                        if exit_code == 0:
                            print(f"[{node_id}] Job finished successfully! Cleaning up TPU...")
                        else:
                            print(f"[{node_id}] Job CRASHED (Exit Code: {exit_code}).")

                        tpu_delete_queued(node_id, zone)
                        active_jobs.remove(job)
                        continue

                    ips = tpu_get_ips(node_id, zone)
                    update_mesh_config(node_id, ips)
                    print(f"[{node_id}] Running mesh setup...")
                    subprocess.run(["mesh", "setup", node_id])

                    full_cmd = f'mesh run {node_id} "{job.cmd}"'
                    log_file = f"log_{node_id}.txt"
                    f = open(log_file, "a", buffering=1)

                    job_processes[node_id] = subprocess.Popen(
                        full_cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, text=True
                    )
                    print(f"[{node_id}] Launched. Monitoring for failure...")
                    print(f"[{node_id}] Started! View logs with: tail -f {log_file}")

            elif state in ["FAILED", "SUSPENDED", "NOT_FOUND"]:
                print(f"[PREEMPTION] {node_id} hardware is {state}. Attempting recovery...")

                if is_running_locally:
                    proc.terminate()

                job_processes[node_id] = None

                if state != "NOT_FOUND":
                    tpu_delete_queued(node_id, zone)
                    time.sleep(15)
                tpu_create_queued(node_id, job.tpu_type, job.runtime, zone, spot=True)

        time.sleep(60)


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
