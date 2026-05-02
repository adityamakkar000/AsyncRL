"""Run launcher for TPU clusters via mesh."""

from __future__ import annotations
import os
import subprocess
import requests

import itertools
import random
from dataclasses import dataclass, field
from typing import Any

from .main import Runtime, TPUJob, TPUType, Zone

# assuming TPU_SERVER_URL is set in the environment
TPU_SERVER_URL = os.getenv("TPU_SERVER_URL", None)


@dataclass
class Vals:
    """Leaf node: a single parameter with a list of candidate values."""

    param: str
    values: list[Any]

    def expand(self) -> list[dict[str, Any]]:
        return [{self.param: v} for v in self.values]


@dataclass
class Cross:
    """Cartesian product of child axes."""

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
    """Lockstep zip of child axes (all children must expand to the same length)."""

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


@dataclass
class LAUNCH_JOB:
    RUN: Cross | Zip | Vals
    EXPERIMENT_PREFIX: str
    FIXED_OVERRIDES: dict[str, Any]
    BASE_CONFIG: str
    ZONE: Zone
    TPU_TYPE: TPUType
    RUNTIME: Runtime
    RETRIES: int = 3


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
    TPU_TYPE: TPUType,
    RUNTIME: Runtime,
    RETRIES: int,
):
    NODE_COUNTER = 0

    for combo in combos:
        NODE_COUNTER += 1
        rng_combo = random.randint(0, 1000000)

        name = make_name(combo, EXPERIMENT_PREFIX)
        overrides = {**FIXED_OVERRIDES, **combo}
        inner_parts = [
            "python",
            "-m",
            "src.train",
            f"--config-name={BASE_CONFIG}",
            f"experiment_name={name}",
        ]
        for k, v in overrides.items():
            inner_parts.append(f"{k}={v}")
        inner_cmd = " ".join(inner_parts)

        # warning: this is hard coded to this project structure
        copy_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        node_id = f"node_{NODE_COUNTER}_{rng_combo}"
        # assuming that server is named "server" in cluster.yaml and ~/.ssh/config
        subprocess.run(["mesh", "copy", "server", f"~/{node_id}"], cwd=copy_dir)

        post_args = {
            "node_id": node_id,
            "zone": ZONE,
            "tpu_type": TPU_TYPE,
            "runtime": RUNTIME,
            "cmd": inner_cmd,
            "retries": RETRIES,
        }

        response = requests.post(f"{TPU_SERVER_URL}/run_job", json=post_args, timeout=30)
        response.raise_for_status()
        print(f"Job submitted: {response.json()}")


def launch(job: LAUNCH_JOB) -> None:
    combos = make_combos(job.RUN)

    run_tpu_jobs(
        combos,
        job.EXPERIMENT_PREFIX,
        job.FIXED_OVERRIDES,
        job.BASE_CONFIG,
        job.ZONE,
        job.TPU_TYPE,
        job.RUNTIME,
        job.RETRIES,
    )
