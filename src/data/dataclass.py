from dataclasses import dataclass
import random
from typing import Optional

from src.data.config import SamplingConfig
from src.data.dataset import OmniMath, load_omni_math


@dataclass
class PromptBatch:
    prompts: list[str]
    examples: list[OmniMath]


class OmniMathPromptDataset:
    def __init__(
        self,
        *,
        examples: Optional[list[OmniMath]] = None,
        sampling: Optional[SamplingConfig] = None,
    ) -> None:
        self.sampling = sampling or SamplingConfig()

        if examples is None:
            examples = load_omni_math(
                subset_fraction=float(self.sampling.subset_fraction),
                subset_seed=int(self.sampling.seed),
                min_difficulty=self.sampling.min_difficulty,
                max_difficulty=self.sampling.max_difficulty,
                domain_contains=self.sampling.domain_contains,
                source_contains=self.sampling.source_contains,
            )

        self.examples = examples
        if not self.examples:
            raise ValueError(
                "No examples found. dataset is probably empty after filtering."
            )

        # TODO: fix the placeholder for filtering logic
        self._rng = random.Random(int(self.sampling.seed))

    def format_prompt(self, ex: OmniMath) -> str:
        pf = self.prompt_format
        parts: list[str] = []
        if pf.system_prompt:
            parts.append(pf.system_prompt.strip())
        user = (pf.user_prefix + ex.problem + pf.user_suffix).strip()
        if pf.include_metadata:
            meta = []
            if ex.difficulty is not None:
                meta.append(f"difficulty={ex.difficulty}")
            if ex.domain:
                meta.append(f"domain={';'.join(ex.domain)}")
            if ex.source:
                meta.append(f"source={ex.source}")
            if meta:
                user = user + "\n\n[metadata] " + " | ".join(meta)
        parts.append(user)
        return "\n\n".join(parts).strip()

    def sample(self, batch_size: int) -> PromptBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        batch_examples = [self.examples[self._rng.randrange(len(self.examples))] for _ in range(batch_size)]
        prompts = [self.format_prompt(ex) for ex in batch_examples]
        return PromptBatch(prompts=prompts, examples=batch_examples)

    def __call__(self, batch_size: int) -> list[str]:
        self.last_batch = self.sample(batch_size)
        return self.last_batch.prompts
