from typing import Any

from src.data.config import DataConfig


class DataLoader:
    def __init__(
        self,
        data_config: DataConfig,
        max_length: int,
        split: str,
    ) -> None:
        self.data_config = data_config
        self.max_length = max_length
        self.split = split
        self._examples, self._prompts, self._answers = self._load_from_gcs()
        self._last_examples = []
        self._current_idx = 0

    def _resolve_gcs_path(self) -> str:
        from src.constants import DATA, GS_BUCKET

        if self.data_config.gcs_path:
            return self.data_config.gcs_path
        return f"{GS_BUCKET}/{DATA}/{self.data_config.name}"

    def _load_from_gcs(self) -> tuple[list[dict], list[str], list[str]]:
        from src.data.utils import load_jsonl_from_gcs

        gs_path = self._resolve_gcs_path()
        prompt_key = self.data_config.prompt_column
        answer_key = self.data_config.answer_column
        rows = load_jsonl_from_gcs(gs_path, prompt_column=prompt_key)
        if not rows:
            raise ValueError(f"No rows with '{prompt_key}' found at {gs_path}")
        prompts = [str(r[prompt_key]).strip() for r in rows]
        answers = [str(r[answer_key]).strip() for r in rows]
        # TODO: need a column to add thinking traces
        return rows, prompts, answers

    def get_prompt(self, batch_size: int) -> list[str]:
        start_idx = self._current_idx
        end_idx = self._current_idx + batch_size
        total = len(self._prompts)
        # so we can cycle through the dataset indefinitely
        indices = [i % total for i in range(start_idx, end_idx)]
        self._current_idx = end_idx % total
        return [self._prompts[i] for i in indices]

    @property
    def last_examples(self) -> list[dict]:
        """Return examples aligned with last batch."""
        return self._last_examples

    def __call__(self, batch_size: int) -> dict[str, Any]:
        start_idx = self._current_idx
        end_idx = self._current_idx + batch_size
        total = len(self._prompts)
        indices = [i % total for i in range(start_idx, end_idx)]
        prompts = [self._prompts[i] for i in indices]
        examples = [self._examples[i] for i in indices]
        self._last_examples = examples
        self._current_idx = end_idx % total
        return {
            "prompts": prompts,
            "examples": examples,
        }

    def save_checkpoint(self) -> dict[str, Any]:
        """Return current index for checkpointing."""
        return {
            "current_idx": self._current_idx,
        }

    def restore_checkpoint(self, state: dict[str, Any]) -> None:
        """Restore index from checkpoint."""
        if "current_idx" not in state:
            raise ValueError("Missing 'current_idx' in checkpoint state")
        self._current_idx = state["current_idx"]
