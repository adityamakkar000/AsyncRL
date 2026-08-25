"""Run launcher for TPU clusters via mesh."""

from __future__ import annotations

import getpass
import itertools
import os
import random
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import requests

from .main import Runtime, TPUType, Zone

TPU_SERVER_URL = os.getenv("TPU_SERVER_URL", None)


@dataclass
class Vals:
    param: str
    values: list[Any]

    def expand(self) -> list[dict[str, Any]]:
        return [{self.param: v} for v in self.values]


@dataclass
class Cross:
    children: list[Vals | Cross | Zip] = field(default_factory=list)

    def expand(self) -> list[dict[str, Any]]:
        if not self.children:
            return [{}]
        child_expansions = [c.expand() for c in self.children]
        combos: list[dict[str, Any]] = []
        for parts in itertools.product(*child_expansions):
            merged: dict[str, Any] = {}
            for d in parts:
                merged.update(d)
            combos.append(merged)
        return combos


@dataclass
class Zip:
    children: list[Vals | Cross | Zip] = field(default_factory=list)

    def expand(self) -> list[dict[str, Any]]:
        if not self.children:
            return [{}]
        child_expansions = [c.expand() for c in self.children]
        lengths = {len(e) for e in child_expansions}
        if len(lengths) != 1:
            raise ValueError(
                f"All children in a Zip must expand to the same length, "
                f"got lengths {[len(e) for e in child_expansions]}"
            )
        combos: list[dict[str, Any]] = []
        for rows in zip(*child_expansions):
            merged: dict[str, Any] = {}
            for d in rows:
                merged.update(d)
            combos.append(merged)
        return combos


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


def make_combos(RUN: Cross | Zip | Vals) -> list[dict[str, Any]]:
    return RUN.expand()


def make_name(combo: dict[str, Any], EXPERIMENT_PREFIX: str) -> str:
    parts = [EXPERIMENT_PREFIX]
    for path, val in combo.items():
        short = path.split(".")[-1][:10]
        parts.append(f"{short}{val:g}" if isinstance(val, float) else f"{short}{val}")
    return "_".join(parts)


def run_tpu_jobs(
    combos: list[dict[str, Any]],
    EXPERIMENT_PREFIX: str,
    FIXED_OVERRIDES: dict[str, Any],
    BASE_CONFIG: str,
    ZONE: Zone,
    JOB_TYPE: JOB_TYPES,
    TPU_TYPE: TPUType,
    RUNTIME: Runtime,
    RETRIES: int,
    keep_logs: bool = True,
    debug: bool = False,
):
    # this get's current project assuming you used launch.py from the project root
    copy_dir = os.getcwd()
    user = getpass.getuser()
    cwd_id = random.randint(0, 1000000)
    server_dir = f"{user}_{EXPERIMENT_PREFIX}_{cwd_id}"

    # NOTE: this assumes the launch server is named "server" in cluster.yaml
    subprocess.run(["mesh", "copy", "server", f"~/{server_dir}"], cwd=copy_dir)

    NODE_COUNTER = 0
    for combo in combos:
        NODE_COUNTER += 1
        rng_combo = random.randint(0, 1000000)
        node_id = f"node_{NODE_COUNTER}_{rng_combo}"

        name = make_name(combo, EXPERIMENT_PREFIX)
        overrides = {**FIXED_OVERRIDES, **combo}
        inner_parts = [
            "python",
            "-m",
            JOB_TYPE.value,
            f"--config-name {BASE_CONFIG}",
        ]
        if JOB_TYPE == JOB_TYPES.TRAIN:
            inner_parts.append(f"experiment_name={name}")

        for k, v in overrides.items():
            inner_parts.append(f"{k}={v}")
        inner_cmd = " ".join(inner_parts)

        post_args = {
            "node_id": node_id,
            "zone": ZONE,
            "tpu_type": TPU_TYPE,
            "runtime": RUNTIME,
            "cmd": inner_cmd,
            "retries": RETRIES,
            "cwd": server_dir,
            "launched_by": user,
            "keep_logs": keep_logs,  # to keep logs after run ends
        }

        if not debug:
            try:
                response = requests.post(f"{TPU_SERVER_URL}/run_job", json=post_args, timeout=30)
                response.raise_for_status()
                print(f"Job submitted: {response.json()}, details: {response.text}")
            except requests.exceptions.HTTPError as err:
                if err.response is not None and err.response.status_code == 409:
                    print("Conflict Details:", err.response.text)  # Look here for the exact cause
                else:
                    print("HTTP Error:", err)
        else:
            print(f"Debug mode enabled. Would have run job with args: {post_args}")


def launch(job: LAUNCH_JOB) -> None:
    combos = make_combos(job.RUN)

    run_tpu_jobs(
        combos,
        job.EXPERIMENT_PREFIX,
        job.FIXED_OVERRIDES,
        job.BASE_CONFIG,
        job.ZONE,
        job.JOB_TYPE,
        job.TPU_TYPE,
        job.RUNTIME,
        job.RETRIES,
        keep_logs=True,
        debug=job.DEBUG,
    )
