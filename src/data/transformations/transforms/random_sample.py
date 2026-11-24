import random
from collections.abc import Callable
from typing import Any


class RandomSample:
    def __init__(self, n: int, seed: int = 0) -> None:
        self.n = n
        self.seed = seed

    def __call__(self, problems: list, question: Callable[[Any], str]) -> list:
        if len(problems) <= self.n:
            return problems
        return random.Random(self.seed).sample(problems, self.n)
