import json
import os
import subprocess

PROJECT = os.getenv("GCLOUD_TPU_PROJECT")


def tpu_describe(node_id, zone):
    """Returns the JSON description of a Queued Resource"""
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "queued-resources",
        "describe",
        node_id,
        f"--zone={zone.value}",
        f"--project={PROJECT}",
        "--format=json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        data = json.loads(res.stdout)
        return data.get("state", {}).get("state", "UNKNOWN")
    return "NOT_FOUND"


def tpu_get_ips(node_id, zone):
    """Fetches External IPs for a TPU"""
    cmd = ["gcloud", "compute", "tpus", "tpu-vm", "describe", node_id, f"--zone={zone.value}", "--format=json"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        data = json.loads(res.stdout)
        return [endpoint["accessConfig"]["externalIp"] for endpoint in data.get("networkEndpoints", [])]
    return []


def tpu_create_queued(node_id, tpu_type, runtime, zone, spot=True):
    """Wraps tpuq_create logic."""
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "queued-resources",
        "create",
        node_id,
        f"--node-id={node_id}",
        f"--zone={zone.value}",
        f"--project={PROJECT}",
        f"--accelerator-type={tpu_type.value}",
        f"--runtime-version={runtime.value}",
    ]
    if spot:
        cmd.append("--spot")
    return subprocess.run(cmd)


def tpu_delete_queued(node_id, zone):
    """Wraps tpuq_rm logic."""
    return subprocess.run(
        [
            "gcloud",
            "compute",
            "tpus",
            "queued-resources",
            "delete",
            node_id,
            f"--zone={zone.value}",
            f"--project={PROJECT}",
            "--force",
            "--quiet",
        ]
    )
