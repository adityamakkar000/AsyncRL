import getpass
import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import requests
from dotenv import load_dotenv

from .sweep import Cross, Vals, Zip
from .tpu import Runtime, TPUType, Zone

load_dotenv()
TPU_SERVER_URL = os.getenv("TPU_SERVER_URL", None)
MAX_TAG_LENGTH = 64
SWEEP_SEPARATOR = "_sweep_"


class JobType(str, Enum):
    TRAIN = "src.train"
    EVAL = "src.eval"


@dataclass
class Job:
    run: Cross | Zip | Vals
    experiment_prefix: str
    fixed_overrides: dict[str, Any]
    base_config: str
    job_type: JobType = JobType.TRAIN


@dataclass
class ResumeJob:
    name: str
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class Target:
    zone: Zone
    tpu_type: TPUType
    runtime: Runtime
    retries: int
    debug: bool


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


def hydra_str(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def short_key(key: str) -> str:
    return re.split(r"[./]", key)[-1]


def notes(combo: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in combo.items())


def tags(job: Job, launch_name: str, index: int, combo: dict[str, Any]) -> list[str]:
    candidates = [
        job.experiment_prefix,
        launch_name,
        f"sweep_{index}",
        *(f"{short_key(k)}={v}" for k, v in combo.items()),
    ]
    return [t for t in candidates if len(t) <= MAX_TAG_LENGTH]


def command(job: Job, run_name: str, launch_name: str, index: int, combo: dict[str, Any]) -> str:
    overrides = {**job.fixed_overrides, **combo}
    if job.job_type == JobType.TRAIN:
        overrides["experiment_name"] = run_name
        overrides["wandb_config.tags"] = (
            "[" + ",".join(hydra_str(t) for t in tags(job, launch_name, index, combo)) + "]"
        )
        if combo:
            overrides["wandb_config.notes"] = hydra_str(notes(combo))
    parts = ["python", "-m", job.job_type.value, "--config-name", job.base_config]
    parts += [shlex.quote(f"{k}={v}") for k, v in overrides.items()]
    return " ".join(parts)


def split_run_name(run_name: str) -> tuple[str, int]:
    launch_name, sep, index = run_name.rpartition(SWEEP_SEPARATOR)
    assert sep and index.isdigit(), f"{run_name} is not a <launch_name>{SWEEP_SEPARATOR}<i> run name"
    return launch_name, int(index)


def copy_code(launch_name: str) -> None:
    copy = subprocess.run(["mesh", "copy", "server", f"~/jobs/{launch_name}"], cwd=os.getcwd(), check=False)
    if copy.returncode != 0:
        raise RuntimeError("mesh copy to the server failed, nothing was submitted")


def stored_command(run_name: str) -> str:
    response = requests.get(f"{TPU_SERVER_URL}/runs/{run_name}/command", timeout=30)
    if response.status_code == 404:
        raise RuntimeError(f"no stored command for {run_name} on the server")
    response.raise_for_status()
    return response.json()["command"]


def post_args(target: Target, run_name: str, cmd: str) -> dict[str, Any]:
    launch_name, _ = split_run_name(run_name)
    return {
        "node_id": run_name,
        "zone": target.zone,
        "tpu_type": target.tpu_type,
        "runtime": target.runtime,
        "cmd": cmd,
        "retries": target.retries,
        "cwd": f"jobs/{launch_name}",
        "launched_by": getpass.getuser(),
    }


def send(target: Target, run_name: str, cmd: str, label: str) -> None:
    args = post_args(target, run_name, cmd)
    result = "debug, not submitted" if target.debug else submit(args)
    print(f"  {run_name}  {label}  -> {result}")
    if target.debug:
        print(f"    {args}")


def launch_new(target: Target, job: Job) -> None:
    launch_name = f"{job.experiment_prefix}_{datetime.now().astimezone():%Y_%m_%d_%H_%M_%S}"
    combos = job.run.expand()
    print(f"\n{launch_name}: {len(combos)} runs")
    if not target.debug:
        copy_code(launch_name)
    for i, combo in enumerate(combos):
        run_name = f"{launch_name}{SWEEP_SEPARATOR}{i}"
        send(target, run_name, command(job, run_name, launch_name, i, combo), notes(combo) or "-")


def launch_resume(target: Target, jobs: list[ResumeJob]) -> None:
    print(f"\nresuming {len(jobs)} runs")
    for job in jobs:
        split_run_name(job.name)
        cmd = " ".join([stored_command(job.name), *(shlex.quote(f"{k}={v}") for k, v in job.overrides.items())])
        send(target, job.name, cmd, f"resume {notes(job.overrides)}".strip())


def launch_job(
    *,
    zone: Zone,
    tpu_type: TPUType,
    runtime: Runtime,
    job: Job | None = None,
    resume_from: list[ResumeJob] | None = None,
    retries: int = 3,
    debug: bool = False,
) -> None:
    assert (job is None) != (not resume_from), "give exactly one of job or resume_from"
    if not TPU_SERVER_URL and not (debug and job is not None):
        raise RuntimeError("TPU_SERVER_URL is not set (check .env)")
    target = Target(zone, tpu_type, runtime, retries, debug)
    if job is not None:
        launch_new(target, job)
    else:
        assert resume_from
        launch_resume(target, resume_from)
