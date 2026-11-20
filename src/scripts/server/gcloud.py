import json
import os
import subprocess

from dotenv import load_dotenv

load_dotenv()
PROJECT = os.getenv("GCLOUD_TPU_PROJECT")
GCLOUD_TIMEOUT = 120


def _gcloud(args: list[str], timeout: int = GCLOUD_TIMEOUT) -> subprocess.CompletedProcess:
    cmd = ["gcloud", "compute", "tpus", *args, f"--project={PROJECT}"]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 1, "", f"timed out after {timeout}s")


def _is_not_found(res: subprocess.CompletedProcess, node_id: str) -> bool:
    return "NOT_FOUND" in res.stderr and f"/{node_id}'" in res.stderr


def error_tail(res: subprocess.CompletedProcess, n: int = 200) -> str:
    return " ".join(res.stderr.split())[-n:]


def tpu_describe(node_id: str, zone: str) -> str:
    res = _gcloud(["queued-resources", "describe", node_id, f"--zone={zone}", "--format=json"], timeout=60)
    if res.returncode == 0:
        return json.loads(res.stdout).get("state", {}).get("state", "UNKNOWN")
    return "NOT_FOUND" if _is_not_found(res, node_id) else "UNKNOWN"


def tpu_node(node_id: str, zone: str) -> tuple[str, list[str]]:
    res = _gcloud(["tpu-vm", "describe", node_id, f"--zone={zone}", "--format=json"], timeout=60)
    if res.returncode != 0:
        return ("NOT_FOUND" if _is_not_found(res, node_id) else "UNKNOWN"), []
    data = json.loads(res.stdout)
    ips = [e["accessConfig"]["externalIp"] for e in data.get("networkEndpoints", []) if e.get("accessConfig")]
    return data.get("state", "UNKNOWN"), ips


def tpu_create_queued(node_id: str, tpu_type: str, runtime: str, zone: str, spot=True) -> subprocess.CompletedProcess:
    args = [
        "queued-resources",
        "create",
        node_id,
        f"--node-id={node_id}",
        f"--zone={zone}",
        f"--accelerator-type={tpu_type}",
        f"--runtime-version={runtime}",
    ]
    if spot:
        args.append("--spot")
    return _gcloud(args)


def tpu_delete_queued(node_id: str, zone: str) -> subprocess.CompletedProcess:
    res = _gcloud(["queued-resources", "delete", node_id, f"--zone={zone}", "--force", "--quiet"], timeout=600)
    if res.returncode != 0 and _is_not_found(res, node_id):
        return subprocess.CompletedProcess(res.args, 0, res.stdout, res.stderr)
    return res
