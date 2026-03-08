"""Hyperparameter sweep launcher for TPU clusters via mesh."""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any


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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TPU_CLUSTERS: list[str] = [
    "node6",
    "node62",
    "node40",
    "node43",
    "node42",
    "node45",
    "node44",
    "node46",
    "node34",
    "node35",
    "node36",
    "node32",
    "node38",
    "node39",
    "node20",
    "node18",
    "node19",
    "node16",
]

BASE_CONFIG: str = "run1"
EXPERIMENT_PREFIX: str = "day3"

lrs = [3e-6, 1e-6, 5e-7]
minbatch_size = [512, 128, 64]
grad_steps = [8, 2, 1]
dataset = ["omnimath_debug_250_500", "omnimath_debug_1500_1750"]
algo = ["cispo"]

SWEEP: Cross | Zip | Vals = Cross(
    [
        Zip(
            [
                Vals("learning_rate_init", lrs),
                Vals("learning_rate_peak", lrs),
                Vals("learning_rate_end", lrs),
            ]
        ),
        Zip(
            [
                Vals("loss_config.rl_config.ppo_minibatch_size", minbatch_size),
                Vals("grad_accum_steps", grad_steps),
            ]
        ),
        Vals("loss_config.rl_config.algorithm", algo),
        Vals("data_config/datasets@data_config.train_config", dataset),
    ]
)

FIXED_OVERRIDES: dict[str, Any] = {
    "wandb_config.project": "AnnealedRL",
}

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def make_combos() -> list[dict[str, Any]]:
    return SWEEP.expand()


def make_name(combo: dict[str, Any]) -> str:
    parts = [EXPERIMENT_PREFIX]
    for path, val in combo.items():
        short = path.split(".")[-1][:10]
        parts.append(f"{short}{val:g}" if isinstance(val, float) else f"{short}{val}")
    name_str = "_".join(parts)
    final_name = str(abs(hash(name_str)))
    return final_name


def mesh_cmd(cluster: str, combo: dict[str, Any]) -> list[str]:
    name = make_name(combo)
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
    cmd = [
        "mesh",
        "run",
        cluster,
        inner_cmd,
    ]
    return cmd


def run_parallel(cmds: list[tuple[str, list[str]]], print_output=False, time_delay: float = 30) -> bool:
    procs: list[tuple[str, subprocess.Popen]] = []
    for i, (name, cmd) in enumerate(cmds):
        procs.append((name, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)))
        print(f"\tLaunched {i + 1}/{len(cmds)}: {name}")
        print(f"\tWaiting {time_delay} seconds before launching next...")
        time.sleep(time_delay)
    ok = True
    for name, p in procs:
        out, _ = p.communicate()
        status = "OK" if p.returncode == 0 else "FAILED"
        print(f"  [{name}] {status}")
        if print_output:
            print(out)
        ok &= p.returncode == 0
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch hparam sweep via mesh")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    combos = make_combos()
    assignments = list(zip(TPU_CLUSTERS, combos))

    print(f"\n{'=' * 60}\nSWEEP: {len(assignments)} run(s)\n{'=' * 60}")
    for cluster, combo in assignments:
        cmd = mesh_cmd(cluster, combo)
        print(f"  [{cluster}] {make_name(combo)}")
        print(f"    {' '.join(cmd)}")
        print(f"    {'-' * 40}")

    if args.debug:
        return

    if len(combos) > len(TPU_CLUSTERS):
        cont = input("WARNING: More combos than clusters, some combos will not be run. Continue? [y/N] ")
        if cont.lower() != "y":
            sys.exit(f"Error: {len(combos)} combos but only {len(TPU_CLUSTERS)} clusters")
        print("Proceeding with launch despite insufficient clusters...")

    print("Setting up clusters...")
    if not run_parallel([(c, ["mesh", "setup", c]) for c, _ in assignments], print_output=False, time_delay=0.1):
        sys.exit("Setup failed")
    print("Launching runs...")
    if not run_parallel([(c, mesh_cmd(c, combo)) for c, combo in assignments], print_output=True, time_delay=30):
        sys.exit("Some runs failed")

    print("All runs completed.")


if __name__ == "__main__":
    main()
