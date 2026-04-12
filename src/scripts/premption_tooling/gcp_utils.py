import json
import os
import subprocess

from dotenv import load_dotenv

load_dotenv()
PROJECT = os.getenv("GCLOUD_TPU_PROJECT")


def tpu_describe(node_id, zone: str):
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "queued-resources",
        "describe",
        node_id,
        f"--zone={zone}",
        f"--project={PROJECT}",
        "--format=json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        data = json.loads(res.stdout)
        return data.get("state", {}).get("state", "UNKNOWN")
    return "NOT_FOUND"


def tpu_get_ips(node_id: str, zone: str) -> list[str]:
    cmd = ["gcloud", "compute", "tpus", "tpu-vm", "describe", node_id, f"--zone={zone}", "--format=json"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        data = json.loads(res.stdout)
        return [endpoint["accessConfig"]["externalIp"] for endpoint in data.get("networkEndpoints", [])]
    return []


def tpu_create_queued(node_id: str, tpu_type: str, runtime: str, zone: str, spot=True) -> subprocess.CompletedProcess:
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "queued-resources",
        "create",
        node_id,
        f"--node-id={node_id}",
        f"--zone={zone}",
        f"--project={PROJECT}",
        f"--accelerator-type={tpu_type}",
        f"--runtime-version={runtime}",
    ]
    if spot:
        cmd.append("--spot")
    return subprocess.run(cmd)


def tpu_delete_queued(node_id, zone: str):
    return subprocess.run(
        [
            "gcloud",
            "compute",
            "tpus",
            "queued-resources",
            "delete",
            node_id,
            f"--zone={zone}",
            f"--project={PROJECT}",
            "--force",
            "--quiet",
        ]
    )
