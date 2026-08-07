import abc
import random
import re
from typing import Any

from .config import Sample
from .utils import get_chat_template

CHINESE_CHARACTERS = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


class Filter(abc.ABC):
    tokenizer: Any = None
    use_system_prompt: bool = False

    def bind(self, tokenizer: Any, use_system_prompt: bool) -> None:
        self.tokenizer = tokenizer
        self.use_system_prompt = use_system_prompt

    def select(self, samples: list[Sample]) -> list[Sample]:
        return [sample for sample in samples if self(sample)]

    @abc.abstractmethod
    def __call__(self, sample: Sample) -> bool:
        raise NotImplementedError()


class PromptLengthFilter(Filter):
    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self, sample: Sample) -> bool:
        tokens = self.tokenizer.apply_chat_template(
            get_chat_template(self.use_system_prompt, sample.prompt),
            add_generation_prompt=True,
            enable_thinking=True,
            tokenize=True,
        )
        return len(tokens) <= self.max_length


class SolutionLengthFilter(Filter):
    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self, sample: Sample) -> bool:
        if sample.solution is None:
            return True
        return len(self.tokenizer.encode(sample.solution, add_special_tokens=True)) <= self.max_length


class EnglishFilter(Filter):
    def __call__(self, sample: Sample) -> bool:
        fields = (sample.prompt, sample.answer, sample.solution)
        return not any(CHINESE_CHARACTERS.search(field) for field in fields if field)


class RandomSampleFilter(Filter):
    def __init__(self, n: int, seed: int = 0):
        self.n = n
        self.seed = seed

    def __call__(self, sample: Sample) -> bool:
        return True

    def select(self, samples: list[Sample]) -> list[Sample]:
        if len(samples) <= self.n:
            return samples
        return random.Random(self.seed).sample(samples, self.n)
