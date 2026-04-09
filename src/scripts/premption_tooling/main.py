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


def run_session(active_jobs: list[TPUJob]):
    print(f"Monitoring {len(active_jobs)} active TPU runs...")

    launched_successfully = {job.node_id: False for job in active_jobs}

    while True:
        print("Checking TPU statuses...")
        for job in active_jobs:
            node_id = job.node_id
            zone = job.zone

            state = tpu_describe(node_id, zone)
            print(f"[{node_id} @ {zone}]: {state}")

            if state == "ACTIVE":
                if not launched_successfully[node_id]:
                    ips = tpu_get_ips(node_id, zone)
                    update_mesh_config(node_id, ips)

                    print(f"[{node_id}] Running mesh setup...")
                    subprocess.run(["mesh", "setup", node_id])

                    full_cmd = f'mesh run {node_id} "{job.cmd}"'
                    log_file = f"log_{node_id}.txt"
                    f = open(log_file, "a", buffering=1)
                    subprocess.Popen(full_cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, text=True)
                    print(f"[{node_id}] Started! View logs with: tail -f {log_file}")

                    launched_successfully[node_id] = True

            elif state in ["FAILED", "SUSPENDED", "NOT_FOUND"]:
                print(f"[RECOVERY] {node_id} is {state}. Re-allocating...")
                launched_successfully[node_id] = False

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
