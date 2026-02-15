"""Base RL dataset interface and implementations.

The dataset is callable: dataset(batch_size) returns a batch for the train step.
For RL, the batch typically contains tokenized prompts. Use get_prompt() for raw
strings when you need them (e.g. for reward computation, logging).
"""

import random
from abc import ABC, abstractmethod
from typing import Any, Protocol

import jax.numpy as jnp

from src.data.config import DataConfig


class RLDataset(Protocol):
    """Protocol for RL datasets. Callable returns batch; checkpoint methods for resumability."""

    def get_prompt(self, batch_size: int) -> list[str]:
        """Return raw prompt strings. Used for generation, reward computation, etc."""
        ...

    def save_checkpoint(self) -> dict[str, Any]:
        """Return pytree to be stored in state['dataset']. Trainer handles the rest."""
        ...

    def restore_checkpoint(self, state: dict[str, Any]) -> None:
        """Restore from pytree. Called with state['dataset'] from checkpoint."""
        ...

    def __call__(self, batch_size: int) -> dict[str, Any]:
        """Return batch for train step. Typically includes tokens, token_mask, prompts."""
        ...


class PromptRLDatasetBase(ABC):
    """Base class for RL datasets that provide prompts. Implements checkpointing."""

    def __init__(self, data_config: DataConfig, seed: int = 0) -> None:
        self.data_config = data_config
        self.seed = seed
        self._rng = random.Random(seed)

    @abstractmethod
    def get_prompt(self, batch_size: int) -> list[str]:
        """Sample and return batch_size prompt strings."""
        ...

    def save_checkpoint(self) -> dict[str, Any]:
        """Return state for checkpoint. Override to add dataset-specific state."""
        return {
            "rng_state": self._rng.getstate(),
            "seed": self.seed,
        }

    def restore_checkpoint(self, state: dict[str, Any]) -> None:
        """Restore from checkpoint state."""
        if "rng_state" in state:
            self._rng.setstate(state["rng_state"])
        if "seed" in state:
            self.seed = state["seed"]


class PromptRLDataset(PromptRLDatasetBase):
    """RL dataset that loads from GCS (processed JSONL from process_datasets).

    GCS path: gs://bucket/data/{dataset_name}/ with 000_*.jsonl, 001_*.jsonl, etc.
    When used as callable: dataset(batch_size) -> dict with keys:
        - prompts: list[str]
        - tokens: jax.Array [batch_size, max_len]
        - token_mask: jax.Array [batch_size, max_len] (1 = real token, 0 = pad)
        - examples: list[dict] aligned with prompts (for reward: answer, solution, etc.)
    """

    def __init__(
        self,
        data_config: DataConfig,
        tokenizer: Any,
        max_length: int = 2048,
        seed: int = 0,
    ) -> None:
        super().__init__(data_config, seed)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._examples, self._prompts = self._load_from_gcs()

    def _resolve_gcs_path(self) -> str:
        from src.constants import DATA, GS_BUCKET

        if self.data_config.gcs_path:
            return self.data_config.gcs_path
        return f"{GS_BUCKET}/{DATA}/{self.data_config.name}"

    def _load_from_gcs(self) -> tuple[list[dict], list[str]]:
        """Load from GCS JSONL files (output of process_datasets)."""
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
        """Examples aligned with last batch (for reward: answer, solution, etc.)."""
        return getattr(self, "_last_examples", [])

    def _set_last_batch(self, prompts: list[str], examples: list[dict]) -> None:
        self._last_examples = examples

    def _tokenize_batch(self, prompts: list[str]) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Tokenize prompts, return (tokens, token_mask)."""
        out = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        tokens = jnp.array(out["input_ids"], dtype=jnp.int32)
        attention_mask = out.get("attention_mask")
        token_mask = (
            jnp.array(attention_mask, dtype=jnp.int32)
            if attention_mask is not None
            else jnp.ones_like(tokens)
        )
        return tokens, token_mask

    def __call__(self, batch_size: int) -> dict[str, Any]:
        indices = [self._rng.randrange(len(self._prompts)) for _ in range(batch_size)]
        prompts = [self._prompts[i] for i in indices]
        examples = [self._examples[i] for i in indices]
        self._set_last_batch(prompts, examples)
        tokens, token_mask = self._tokenize_batch(prompts)
        return {
            "prompts": prompts,
            "tokens": tokens,
            "token_mask": token_mask,
            "examples": examples,
        }
