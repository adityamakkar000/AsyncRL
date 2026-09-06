import abc
import random
import re
from dataclasses import replace

from .config import Sample

ESCAPED_LATEX = [
    (re.compile(r"\x0c(?=rac|loor|orall|box|rown)"), r"\\f"),
    (re.compile(r"\x0c"), ""),
    (re.compile(r"\x0b(?=ec|arphi|artheta|artriangle|arepsilon|ert|dots|space)"), r"\\v"),
    (re.compile(r"\x0b"), ""),
    (re.compile(r"\t(?=heta|herefore|an\b|anh|au\b|ext|frac|imes|o\b|op\b|riangle)"), r"\\t"),
]


class Transform(abc.ABC):
    @abc.abstractmethod
    def __call__(self, samples: list[Sample]) -> tuple[list[Sample], int]:
        raise NotImplementedError()


class FixEscapedLatex(Transform):
    def __call__(self, samples: list[Sample]) -> tuple[list[Sample], int]:
        def fix(text: str | None) -> str | None:
            if text is None:
                return None
            for pattern, repl in ESCAPED_LATEX:
                text = pattern.sub(repl, text)
            return text

        fixed = [replace(s, prompt=fix(s.prompt), solution=fix(s.solution)) for s in samples]
        n_changed = sum(a.prompt != b.prompt or a.solution != b.solution for a, b in zip(samples, fixed))
        return fixed, n_changed


class Dedupe(Transform):
    def __call__(self, samples: list[Sample]) -> tuple[list[Sample], int]:
        seen: set[str] = set()
        kept = []
        for sample in samples:
            if sample.prompt not in seen:
                seen.add(sample.prompt)
                kept.append(sample)
        return kept, len(samples) - len(kept)


class RandomSample(Transform):
    def __init__(self, n: int, seed: int = 0):
        self.n = n
        self.seed = seed

    def __call__(self, samples: list[Sample]) -> tuple[list[Sample], int]:
        kept = samples if len(samples) <= self.n else random.Random(self.seed).sample(samples, self.n)
        return kept, len(kept)
