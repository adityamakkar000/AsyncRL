"""Hyperparameter sweep launcher for TPU clusters via mesh."""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
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
    "node9",
    "node8",
    "node7",
    "node6",
    "node12",
    "node13",
    "node10",
    "node11",
]

BASE_CONFIG: str = "run1"
EXPERIMENT_PREFIX: str = "sweep"

lrs = [1e-4, 1e-5, 1e-6, 1e-7]

SWEEP: Cross | Zip | Vals = Cross(
    [
        Zip(
            [
                Vals("learning_rate_init", lrs),
                Vals("learning_rate_peak", lrs),
                Vals("learning_rate_end", lrs),
            ]
        ),
        Vals("loss_config.rl_config.algorithm", ["cispo", "rloo"]),
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
        short = path.split(".")[-1]
        parts.append(f"{short}{val:g}" if isinstance(val, float) else f"{short}{val}")
    return "_".join(parts)


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


def run_parallel(cmds: list[tuple[str, list[str]]]) -> bool:
    procs = [
        (name, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)) for name, cmd in cmds
    ]
    ok = True
    for name, p in procs:
        out, _ = p.communicate()
        status = "OK" if p.returncode == 0 else "FAILED"
        print(f"  [{name}] {status}")
        ok &= p.returncode == 0
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch hparam sweep via mesh")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    combos = make_combos()
    if len(combos) > len(TPU_CLUSTERS):
        sys.exit(f"Error: {len(combos)} combos but only {len(TPU_CLUSTERS)} clusters")

    assignments = list(zip(TPU_CLUSTERS, combos))

    print(f"\n{'=' * 60}\nSWEEP: {len(assignments)} run(s)\n{'=' * 60}")
    for cluster, combo in assignments:
        cmd = mesh_cmd(cluster, combo)
        print(f"  [{cluster}] {make_name(combo)}")
        print(f"    {' '.join(cmd)}")
        print(f"    {'-' * 40}")

    if args.debug:
        return

    print("Setting up clusters...")
    if not run_parallel([(c, ["mesh", "setup", c]) for c, _ in assignments]):
        sys.exit("Setup failed")
    print("Launching runs...")
    if not run_parallel([(c, mesh_cmd(c, combo)) for c, combo in assignments]):
        sys.exit("Some runs failed")

    print("All runs completed.")


if __name__ == "__main__":
    main()
