from dataclasses import dataclass
import random
from typing import Any, Optional

# from src.data.config import DataConfig -- remove when not testing
from config import DataConfig


@dataclass()
class OmniMath:

    id: int
    problem: str
    answer: str
    solution: Optional[str]
    difficulty: Optional[float]
    domain: list[str]
    source: Optional[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "problem": self.problem,
            "answer": self.answer,
            "solution": self.solution,
            "difficulty": self.difficulty,
            "domain": self.domain,
            "source": self.source,
        }


@dataclass
class PromptBatch:
    prompts: list[str]
    examples: list[OmniMath]


class OmniMathPromptDataset:
    """Callable dataset that returns a list of prompts.

    RL-only stage-1: prompts are just the raw problem statement. We keep the
    sampled `examples` aligned with prompts in `last_batch` so reward code can
    use `answer`, etc. later.
    """

    def __init__(
        self,
        *,
        examples: Optional[list[OmniMath]] = None,
        data_config: DataConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.data_config = data_config

        if examples is None:
            examples = load_omni_math(subset_fraction=1.0, subset_seed=int(seed))

        self.examples = examples
        if not self.examples:
            raise ValueError(
                "No examples found. dataset is probably empty after filtering."
            )

        self._rng = random.Random(int(seed))
        self.last_batch: Optional[PromptBatch] = None

    @property
    def last_examples(self) -> list[OmniMath]:
        if self.last_batch is None:
            return []
        return self.last_batch.examples

    def format_prompt(self, ex: OmniMath) -> str:
        return ex.problem.strip()

    def sample(self, batch_size: int) -> PromptBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        batch_examples = [
            self.examples[self._rng.randrange(len(self.examples))]
            for _ in range(batch_size)
        ]
        prompts = [self.format_prompt(ex) for ex in batch_examples]
        return PromptBatch(prompts=prompts, examples=batch_examples)

    def __call__(self, batch_size: int) -> list[str]:
        self.last_batch = self.sample(batch_size)
        return self.last_batch.prompts


def _require_datasets() -> Any:
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing optional dependency `datasets`.\n"
            "Install via: `uv sync --extra data` (see `pyproject.toml`)."
        ) from e
    return load_dataset


def _as_list_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _apply_random_subset(
    examples: list[OmniMath], *, fraction: float, seed: int
) -> list[OmniMath]:
    """Deterministically shuffle + truncate to a subset of examples."""
    if fraction >= 1.0:
        return examples
    if fraction <= 0.0:
        raise ValueError("subset_fraction must be in (0, 1].")
    if not examples:
        return examples

    rng = random.Random(int(seed))
    shuffled = list(examples)
    rng.shuffle(shuffled)
    k = int(len(shuffled) * float(fraction))
    k = max(1, k)
    return shuffled[:k]


def load_omni_math(
    *,
    name: str = "KbsdJames/Omni-MATH",
    split: str = "test",
    cache_dir: Optional[str] = None,
    subset_fraction: float = 1.0,
    subset_seed: int = 0,
    min_difficulty: Optional[float] = None,
    max_difficulty: Optional[float] = None,
    domain_contains: Optional[str] = None,
    source_contains: Optional[str] = None,
) -> list[OmniMath]:

    load_dataset = _require_datasets()
    ds = load_dataset(name, split=split, cache_dir=cache_dir)

    examples: list[OmniMath] = []
    for i, row in enumerate(ds):
        problem = str(row.get("problem", "")).strip()
        answer = str(row.get("answer", "")).strip()
        solution = row.get("solution")
        solution = None if solution is None else str(solution).strip()

        difficulty_raw = row.get("difficulty")
        try:
            difficulty = None if difficulty_raw is None else float(difficulty_raw)
        except (TypeError, ValueError):
            difficulty = None

        domain = _as_list_str(row.get("domain"))
        source = row.get("source")
        source = None if source is None else str(source)

        ex = OmniMath(
            id=i,
            problem=problem,
            answer=answer,
            solution=solution,
            difficulty=difficulty,
            domain=domain,
            source=source,
        )

        if (
            min_difficulty is not None
            and ex.difficulty is not None
            and ex.difficulty < min_difficulty
        ):
            continue
        if (
            max_difficulty is not None
            and ex.difficulty is not None
            and ex.difficulty > max_difficulty
        ):
            continue
        if domain_contains is not None:
            dom = " | ".join(ex.domain).lower()
            if domain_contains.lower() not in dom:
                continue
        if source_contains is not None:
            if (ex.source or "").lower().find(source_contains.lower()) == -1:
                continue

        if not ex.problem:
            continue

        examples.append(ex)

    examples = _apply_random_subset(
        examples, fraction=subset_fraction, seed=subset_seed
    )

    if not examples:
        raise ValueError("Omni-MATH load produced 0 examples after filtering.")

    return examples
