from __future__ import annotations

import getpass
import os
import random
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Any

import requests
from dotenv import load_dotenv

from .sweep import Cross, Vals, Zip
from .tpu import Runtime, TPUType, Zone

load_dotenv()
TPU_SERVER_URL = os.getenv("TPU_SERVER_URL", None)


class JOB_TYPES(str, Enum):
    TRAIN = "src.train"
    EVAL = "src.eval"


@dataclass
class LAUNCH_JOB:
    RUN: Cross | Zip | Vals
    EXPERIMENT_PREFIX: str
    FIXED_OVERRIDES: dict[str, Any]
    BASE_CONFIG: str
    ZONE: Zone
    JOB_TYPE: JOB_TYPES
    TPU_TYPE: TPUType
    RUNTIME: Runtime
    RETRIES: int = 3
    DEBUG: bool = False
    LAUNCH_ID: int | None = None
    SWEEPS: list[int] | None = None
    FRESH_CODE: bool = False


def submit(post_args: dict[str, Any]) -> str:
    try:
        response = requests.post(f"{TPU_SERVER_URL}/run_job", json=post_args, timeout=30)
    except requests.exceptions.RequestException as err:
        return f"server unreachable: {err}"
    if response.status_code == 409:
        return f"skipped: {response.json().get('detail')}"
    if not response.ok:
        return f"rejected ({response.status_code}): {response.text}"
    return "resubmitted (replaced finished run)" if response.json().get("replaced") else "submitted"


def command(job: LAUNCH_JOB, run_id: str, combo: dict[str, Any]) -> str:
    parts = ["python", "-m", job.JOB_TYPE.value, f"--config-name {job.BASE_CONFIG}"]
    if job.JOB_TYPE == JOB_TYPES.TRAIN:
        parts.append(f"experiment_name={run_id}")
    parts += [f"{k}={v}" for k, v in {**job.FIXED_OVERRIDES, **combo}.items()]
    return " ".join(parts)


def code_exists(name: str) -> bool:
    response = requests.get(f"{TPU_SERVER_URL}/code/{name}", timeout=30)
    response.raise_for_status()
    return response.json()["exists"]


def copy_code(job: LAUNCH_JOB, launch_id: int) -> str:
    name = f"{job.EXPERIMENT_PREFIX}_{launch_id}"
    run_dir = f"jobs/{name}"
    if job.DEBUG:
        return run_dir
    if not job.FRESH_CODE and code_exists(name):
        print(f"reusing code already on the server at ~/{run_dir} (set FRESH_CODE=True to replace it)")
        return run_dir
    copy = subprocess.run(["mesh", "copy", "server", f"~/{run_dir}"], cwd=os.getcwd(), check=False)
    if copy.returncode != 0:
        raise RuntimeError("mesh copy to the server failed, nothing was submitted")
    return run_dir


def launch(job: LAUNCH_JOB) -> None:
    combos = job.RUN.expand()
    sweeps = list(range(len(combos))) if job.SWEEPS is None else job.SWEEPS
    bad = [i for i in sweeps if not 0 <= i < len(combos)]
    if bad:
        raise ValueError(f"SWEEPS {bad} out of range, this launch has {len(combos)} runs (0..{len(combos) - 1})")
    if not job.DEBUG and not TPU_SERVER_URL:
        raise RuntimeError("TPU_SERVER_URL is not set (check .env)")

    launch_id = random.randint(0, 1000000) if job.LAUNCH_ID is None else job.LAUNCH_ID
    run_dir = copy_code(job, launch_id)
    user = getpass.getuser()
    print(f"\n{job.EXPERIMENT_PREFIX}: launch_id={launch_id} (relaunch with LAUNCH_ID={launch_id})")
    for i in sweeps:
        combo = combos[i]
        run_id = f"{job.EXPERIMENT_PREFIX}_sweep_{i}_{launch_id}"
        post_args = {
            "node_id": run_id,
            "zone": job.ZONE,
            "tpu_type": job.TPU_TYPE,
            "runtime": job.RUNTIME,
            "cmd": command(job, run_id, combo),
            "retries": job.RETRIES,
            "cwd": run_dir,
            "launched_by": user,
        }
        result = "debug, not submitted" if job.DEBUG else submit(post_args)
        changed = " ".join(f"{k}={v}" for k, v in combo.items()) or "-"
        print(f"  {run_id}  {changed}  -> {result}")
        if job.DEBUG:
            print(f"    {post_args}")
