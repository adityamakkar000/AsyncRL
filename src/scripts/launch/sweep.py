from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any


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
