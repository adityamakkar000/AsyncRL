"""Run launcher for TPU clusters via mesh."""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .main import Runtime, TPUJob, TPUType, Zone


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


class JOB_TYPE(str, Enum):
    TRAIN = "src.train"
    EVAL = "src.eval"


@dataclass
class LaunchJob:
    run: Cross | Zip | Vals
    job_type: JOB_TYPE
    experiment_prefix: str
    fixed_overrides: dict[str, Any]
    base_config: str
    zone: Zone
    tpu_type: TPUType
    runtime: Runtime
    retries: int = 3


def make_combos(run: Cross | Zip | Vals) -> list[dict[str, Any]]:
    return run.expand()


def make_name(combo: dict[str, Any], experiment_prefix: str) -> str:
    parts = [experiment_prefix]
    for path, val in combo.items():
        short = path.split(".")[-1][:10]
        parts.append(f"{short}{val:g}" if isinstance(val, float) else f"{short}{val}")
    return "_".join(parts)


def return_tpu_jobs(
    job: LaunchJob,
) -> list[TPUJob]:
    node_counter = 0

    jobs = []
    for combo in make_combos(job.run):
        node_counter += 1
        rng_combo = random.randint(0, 100000)

        name = make_name(combo, job.experiment_prefix)
        overrides = {**job.fixed_overrides, **combo}
        inner_parts = [
            "python",
            "-m",
            job.job_type.value,
            f"--config-name={job.base_config}",
            f"experiment_name={name}",
        ]
        for k, v in overrides.items():
            inner_parts.append(f"{k}={v}")
        inner_cmd = " ".join(inner_parts)
        # job_tpu = TPUJob(
        #     node_id=f"node_{node_counter}_{rng_combo}_{datatime}",
        #     zone=job.zone,
        #     tpu_type=job.tpu_type,
        #     runtime=job.runtime,
        #     cmd=inner_cmd,
        #     retries=job.retries,
        # )
        # jobs.append(job_tpu)

        subprocess.Run('mesh run mac "mv ~/job ~/{name}"')
        # make post to the server with this info

    return jobs


def launch(job: LaunchJob) -> None:
    tpu_jobs = return_tpu_jobs(job)
    # run_session(tpu_jobs)
    # make
    print("All runs completed.")
