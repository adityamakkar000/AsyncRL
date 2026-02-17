import random
from typing import Any

from src.data.config import DataConfig

class DataLoader:

    def __init__(
        self,
        data_config: DataConfig,
        seed: int = 0,
        max_length: int = 2048, # TODO: @adityamakkar000 add max length
    ) -> None:
        self.data_config = data_config
        self.seed = seed
        self.max_length = max_length
        self._rng = random.Random(seed)
        self._examples, self._prompts = self._load_from_gcs()
        self._last_examples = []

    def _resolve_gcs_path(self) -> str:
        from src.constants import DATA, GS_BUCKET

        if self.data_config.gcs_path:
            return self.data_config.gcs_path
        return f"{GS_BUCKET}/{DATA}/{self.data_config.name}"

    def _load_from_gcs(self) -> tuple[list[dict], list[str]]:
        from src.data.utils import load_jsonl_from_gcs

        gs_path = self._resolve_gcs_path()
        prompt_key = self.data_config.prompt_column
        rows = load_jsonl_from_gcs(gs_path, prompt_column=prompt_key)
        if not rows:
            raise ValueError(f"No rows with '{prompt_key}' found at {gs_path}")
        prompts = [str(r[prompt_key]).strip() for r in rows]
        return rows, prompts

    def get_prompt(self, batch_size: int) -> list[str]:
        indices = [self._rng.randrange(len(self._prompts)) for _ in range(batch_size)]
        return [self._prompts[i] for i in indices]

    @property
    def last_examples(self) -> list[dict]:
        """Return examples aligned with last batch."""
        return self._last_examples

    def __call__(self, batch_size: int) -> dict[str, Any]:
        indices = [self._rng.randrange(len(self._prompts)) for _ in range(batch_size)]
        prompts = [self._prompts[i] for i in indices]
        examples = [self._examples[i] for i in indices]
        self._last_examples = examples
        return {
            "prompts": prompts,
            "examples": examples,
        }

    def save_checkpoint(self) -> dict[str, Any]:
        """Return random generator state and seed for checkpointing."""
        return {
            "rng_state": self._rng.getstate(),
            "seed": self.seed,
        }

    def restore_checkpoint(self, state: dict[str, Any]) -> None:
        """Restore random generator state and seed from checkpoint."""
        if "rng_state" in state:
            self._rng.setstate(state["rng_state"])
        if "seed" in state:
            self.seed = state["seed"]