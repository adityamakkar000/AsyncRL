from collections.abc import Callable
from typing import Any


class Dedupe:
    def __call__(self, problems: list, question: Callable[[Any], str]) -> list:
        seen: set[str] = set()
        kept = []
        for problem in problems:
            if question(problem) not in seen:
                seen.add(question(problem))
                kept.append(problem)
        return kept
