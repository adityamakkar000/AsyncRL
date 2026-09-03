from collections.abc import Callable

from datasets import load_dataset

from src.data.config import Sample

# Registry: name -> function that returns list[Sample] and takes no inputs
GLOBAL_DICT: dict[str, Callable[[], list[Sample]]] = {}


def register_dataset(name: str) -> Callable[[Callable[[], list[Sample]]], Callable[[], list[Sample]]]:
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


@register_dataset("gsm8k")
def load_gsm8k() -> list[Sample]:
    ds = load_dataset("openai/gsm8k", "main", split="train")

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


@register_dataset("gsm8k_val")
def load_gsm8k_val() -> list[Sample]:
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


@register_dataset("amc_25")
def load_amc_25() -> list[Sample]:
    ds = load_dataset("sonthenguyen/amc12-2025-non-figure")["train"]
    samples = [Sample(prompt=example["question"], answer=str(example["answer"]), solution=None) for example in ds]
    ds.cleanup_cache_files()
    return samples


@register_dataset("polaris")
def load_polaris() -> list[Sample]:
    ds = load_dataset("POLARIS-Project/Polaris-Dataset-53K")["train"]
    samples = [Sample(prompt=example["problem"], answer=str(example["answer"]), solution=None) for example in ds]
    ds.cleanup_cache_files()
    return samples


@register_dataset("troll-17k-train")
def load_troll_17k() -> list[Sample]:
    ds = load_dataset("philippbecker/troll_data")["train"]
    TROLL_SUFFIX = "\n\nPresent the answer in LaTeX format: \\boxed{Your answer}."
    samples = [
        Sample(
            prompt=example["prompt"][1]["content"].removesuffix(TROLL_SUFFIX),
            answer=str(example["reward_model"]["ground_truth"]),
            solution=None,
        )
        for example in ds
    ]
    ds.cleanup_cache_files()
    return samples


@register_dataset("troll-10k-test")
def load_troll_10k_test() -> list[Sample]:
    ds = load_dataset("philippbecker/troll_data")["test"]
    samples = [
        Sample(
            prompt=example["prompt"][1]["content"], answer=str(example["reward_model"]["ground_truth"]), solution=None
        )
        for example in ds
    ]
    ds.cleanup_cache_files()
    return samples


@register_dataset("dapo_math_17k")
def load_dapo_math_17k() -> list[Sample]:
    ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
    DAPO_PROMPT_PREFIX = (
        "Solve the following math problem step by step. The last line of your response should be of the form "
        "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
    )
    samples = [
        Sample(
            prompt=example["prompt"][0]["content"].removeprefix(DAPO_PROMPT_PREFIX),
            answer=str(example["reward_model"]["ground_truth"]),
            solution=None,
        )
        for example in ds
    ]
    ds.cleanup_cache_files()
    return samples


if __name__ == "__main__":
    data = None
    while not data:
        inp = input("which dataset do you want to see?").strip()
        data = GLOBAL_DICT.get(inp)
        if data is None:
            continue
        data = data()
    breakpoint()
