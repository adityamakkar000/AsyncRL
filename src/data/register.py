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


@register_dataset("omnimath")
def load_omnimath() -> list[Sample]:
    ds = load_dataset("KbsdJames/Omni-MATH", split="test")
    samples = [
        Sample(prompt=example["problem"], answer=example["answer"], solution=example["solution"]) for example in ds
    ]
    ds.cleanup_cache_files()
    return samples


@register_dataset("omnimath_debug_1k")
def load_omnimath_debug_1k() -> list[Sample]:
    ds = load_dataset("KbsdJames/Omni-MATH", split="test")
    ds_sorted = ds.sort("difficulty")[2000:3000]

    samples = [
        Sample(prompt=ds_sorted["problem"][i], answer=ds_sorted["answer"][i], solution=ds_sorted["solution"][i])
        for i in range(1000)
    ]
    ds.cleanup_cache_files()
    return samples


@register_dataset("omnimath_debug_250_500")
def load_omnimath_debug_250_500() -> list[Sample]:
    """omnimath with the 250-500 problems sorted by difficulty"""

    ds = load_dataset("KbsdJames/Omni-MATH", split="test")
    ds_sorted = ds.sort("difficulty")[250:500]

    samples = [
        Sample(prompt=ds_sorted["problem"][i], answer=ds_sorted["answer"][i], solution=ds_sorted["solution"][i])
        for i in range(250)
    ]
    ds.cleanup_cache_files()
    return samples


@register_dataset("omnimath_debug_500_750")
def load_omnimath_debug_500_750() -> list[Sample]:
    """omnimath with the 500-750 problems sorted by difficulty"""

    ds = load_dataset("KbsdJames/Omni-MATH", split="test")
    ds_sorted = ds.sort("difficulty")[500:750]

    samples = [
        Sample(prompt=ds_sorted["problem"][i], answer=ds_sorted["answer"][i], solution=ds_sorted["solution"][i])
        for i in range(250)
    ]
    ds.cleanup_cache_files()
    return samples


@register_dataset("omnimath_debug_1500_1750")
def load_omnimath_debug_1500_1750() -> list[Sample]:
    """omnimath with the 1500-1750 problems sorted by difficulty"""

    ds = load_dataset("KbsdJames/Omni-MATH", split="test")
    ds_sorted = ds.sort("difficulty")[1500:1750]

    samples = [
        Sample(prompt=ds_sorted["problem"][i], answer=ds_sorted["answer"][i], solution=ds_sorted["solution"][i])
        for i in range(250)
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

    ds.cleanup_cache_files()
    return samples


@register_dataset("gsm8k_hard_256")
def load_gsm8k_hard_256() -> list[Sample]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.add_column("len_solution", [len(solution) for solution in ds["answer"]])
    ds_sorted = ds.sort("len_solution", reverse=True)

    def get_answer(answer: str) -> str:
        """
        Answers are of the form '... #### 15'
        So we seperate, take the last part after the # and strip whitespace to get the answer.
        """
        return answer.split("#")[-1].strip()

    samples = [
        Sample(
            prompt=ds_sorted["question"][i], answer=get_answer(ds_sorted["answer"][i]), solution=ds_sorted["answer"][i]
        )
        for i in range(256)
    ]
    ds.cleanup_cache_files()
    return samples


@register_dataset("aime")
def load_aime() -> list[Sample]:
    ds_2024 = load_dataset("Maxwell-Jia/AIME_2024")["train"]
    aime_2024 = [
        Sample(prompt=example["Problem"], answer=str(example["Answer"]), solution=example["Solution"])
        for example in ds_2024
    ]
    ds_2025 = load_dataset("MathArena/aime_2025")["train"]
    aime_2025 = [Sample(prompt=example["problem"], answer=str(example["answer"]), solution=None) for example in ds_2025]
    ds_2025.cleanup_cache_files()
    return aime_2024 + aime_2025


@register_dataset("aime_2024")
def load_aime_2024() -> list[Sample]:
    ds = load_dataset("Maxwell-Jia/AIME_2024")["train"]
    samples = [
        Sample(prompt=example["Problem"], answer=str(example["Answer"]), solution=example["Solution"]) for example in ds
    ]
    ds.cleanup_cache_files()
    return samples


@register_dataset("aime_2025")
def load_aime_2025() -> list[Sample]:
    ds = load_dataset("MathArena/aime_2025")["train"]
    samples = [Sample(prompt=example["problem"], answer=str(example["answer"]), solution=None) for example in ds]
    ds.cleanup_cache_files()
    return samples


@register_dataset("aime_2026")
def load_aime_2026() -> list[Sample]:
    ds = load_dataset("MathArena/aime_2026")["train"]
    samples = [Sample(prompt=example["problem"], answer=str(example["answer"]), solution=None) for example in ds]
    ds.cleanup_cache_files()
    return samples


@register_dataset("math_500")
def load_math_500() -> list[Sample]:
    ds = load_dataset("HuggingFaceH4/MATH-500")["test"]
    samples = [
        Sample(prompt=example["problem"], answer=example["answer"], solution=example["solution"]) for example in ds
    ]
    ds.cleanup_cache_files()
    return samples


@register_dataset("amc_23")
def load_amc_23() -> list[Sample]:
    ds = load_dataset("zwhe99/amc23")["test"]
    samples = [Sample(prompt=example["question"], answer=str(example["answer"]), solution=None) for example in ds]
    ds.cleanup_cache_files()
    return samples


@register_dataset("polaris")
def load_polaris() -> list[Sample]:
    ds = load_dataset("POLARIS-Project/Polaris-Dataset-53K")["train"]
    samples = [Sample(prompt=example["problem"], answer=str(example["answer"]), solution=None) for example in ds]
    ds.cleanup_cache_files()
    return samples
