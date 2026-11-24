from collections.abc import Callable
from dataclasses import dataclass

from datasets import load_dataset

from ...config import Score
from ..base import Environment, register_dataset
from .math_utils import extract_answer, grade_answer_verl
from .prompts import SYSTEM_PROMPT, apply_prompt_template


@dataclass(frozen=True)
class MathProblem:
    question: str
    answer: str
    solution: str | None = None


Builder = Callable[[dict], MathProblem]


def fields(question: str = "problem", answer: str = "answer", solution: str | None = None) -> Builder:
    return lambda row: MathProblem(row[question], str(row[answer]), row[solution] if solution else None)


def troll(row: dict) -> MathProblem:
    TROLL_SUFFIX = "\n\nPresent the answer in LaTeX format: \\boxed{Your answer}."
    question = row["prompt"][1]["content"].removesuffix(TROLL_SUFFIX)
    return MathProblem(question, str(row["reward_model"]["ground_truth"]))


def gsm8k(row: dict) -> MathProblem:
    return MathProblem(row["question"], row["answer"].split("#")[-1].strip(), row["answer"])


def load(build: Builder, path: str, split: str = "train", config: str | None = None) -> list[MathProblem]:
    ds = load_dataset(path, config, split=split)
    problems = [build(row) for row in ds]
    ds.cleanup_cache_files()
    return problems


class MathEnv(Environment[MathProblem]):
    name = "math"
    system_prompt = SYSTEM_PROMPT

    def question(self, problem: MathProblem) -> str:
        return problem.question

    def user_prompt(self, question: str) -> str:
        return apply_prompt_template(question)

    def score(self, problem: MathProblem, completion: str) -> Score:
        response = completion.rpartition("</think>")[2]
        reward = grade_answer_verl(response, problem.answer)
        extracted = extract_answer(response)
        if reward is None:
            return Score(0.0, "unparsable", extracted, problem.answer)
        return Score(float(reward), "correct" if reward else "wrong", extracted, problem.answer)

    @register_dataset("gsm8k")
    def load_gsm8k() -> list[MathProblem]:
        return load(gsm8k, "openai/gsm8k", "train", "main")

    @register_dataset("gsm8k_val")
    def load_gsm8k_val() -> list[MathProblem]:
        return load(gsm8k, "openai/gsm8k", "test", "main")

    @register_dataset("aime_2025")
    def load_aime_2025() -> list[MathProblem]:
        return load(fields(), "MathArena/aime_2025")

    @register_dataset("aime_2026")
    def load_aime_2026() -> list[MathProblem]:
        return load(fields(), "MathArena/aime_2026")

    @register_dataset("math_500")
    def load_math_500() -> list[MathProblem]:
        return load(fields(solution="solution"), "HuggingFaceH4/MATH-500", "test")

    @register_dataset("amc_25")
    def load_amc_25() -> list[MathProblem]:
        return load(fields("question"), "sonthenguyen/amc12-2025-non-figure")

    @register_dataset("troll-17k-train")
    def load_troll_17k() -> list[MathProblem]:
        return load(troll, "philippbecker/troll_data")

    @register_dataset("troll-10k-test")
    def load_troll_10k_test() -> list[MathProblem]:
        return load(troll, "philippbecker/troll_data", "test")
