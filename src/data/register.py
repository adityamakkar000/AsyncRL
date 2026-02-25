"""
Register datasets here. Each dataset is a function that returns a list of Sample.
Use @register_dataset("name") so the dataset can be created in process.

Recommended approach:
- Use load_dataset from datasets to load the dataset and convert to json string for each row.
- Use Sample.from_json to convert the dataset to a list of Sample.
- Call ds.cleanup_cache_files() to clean up the cache files.

Helper functions:
- load_dataset(name: str, split: str) -> Dataset
- Sample.from_dict(data: dict) -> Sample
"""

from typing import Callable

from datasets import load_dataset

from src.data.config import Sample

# Registry: name -> function that returns list[Sample] and takes no inputs
GLOBAL_DICT: dict[str, Callable[[], list[Sample]]] = {}


def register_dataset(name: str) -> Callable[[Callable[[], list[Sample]]], Callable[[], list[Sample]]]:
    """Decorator to register a dataset function. The function should return a list of Sample and take no inputs."""

    def decorator(fn: Callable[[], list[Sample]]) -> Callable[[], list[Sample]]:
        GLOBAL_DICT[name] = fn
        return fn

    return decorator


# -- Register your dataset below --


@register_dataset("omnimath")
def load_omnimath() -> list[Sample]:
    ds = load_dataset("KbsdJames/Omni-MATH", split="test")
    samples = [
        Sample(prompt=example["problem"], answer=example["answer"], solution=example["solution"]) for example in ds
    ]
    ds.cleanup_cache_files()
    return samples


@register_dataset("gsm8k")
def load_gsm8k() -> list[Sample]:
    ds = load_dataset("openai/gsm8k", "main", split="test")

    def get_answer(answer: str) -> str:
        """
        Answers are of the form '... #### 15'
        So we seperate, take the last part after the # and strip whitespace to get the answer.
        """
        return answer.split("#")[-1].strip()

    samples = [
        Sample(prompt=example["question"], answer=get_answer(example["answer"]), solution=example["answer"])
        for example in ds
    ]

    return samples


@register_dataset("aime")
def load_aime() -> list[Sample]:
    ds_2024 = load_dataset("Maxwell-Jia/AIME_2024")["train"]
    aime_2024 = [
        Sample(prompt=example["Problem"], answer=example["Answer"], solution=example["Solution"]) for example in ds_2024
    ]
    ds_2025 = load_dataset("MathArena/aime_2025")["train"]
    aime_2025 = [Sample(prompt=example["problem"], answer=example["answer"], solution=None) for example in ds_2025]
    return aime_2024 + aime_2025
