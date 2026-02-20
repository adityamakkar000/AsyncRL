from typing import Any
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

from src.constants import DATA, GS_BUCKET
from src.data.config import DatasetConfig, Sample, RLBatch
from src.data.utils import load_jsonl_from_gcs
from src.inference_engine.config import InferenceResults, InferenceRollout
from src.data.verifier import Verifier, VerifierInput


class DataLoader:
    def __init__(
        self,
        dataset_config: DatasetConfig,
        seq_length: int,
        hf_model: str,
    ) -> None:
        self.dataset_config = dataset_config
        self.seq_length = seq_length
        self.hf_model = hf_model
        self.samples = self._load_from_gcs()
        self._last_samples = []
        self._current_idx = 0
        self.verifier = Verifier()

    def _resolve_gcs_path(self) -> str:
        if self.dataset_config.gcs_path:
            return self.dataset_config.gcs_path
        return f"{GS_BUCKET}/{DATA}/{self.dataset_config.name}"

    def _load_from_gcs(self) -> list[Sample]:
        gs_path = self._resolve_gcs_path()
        rows = load_jsonl_from_gcs(gs_path)
        if not rows:
            raise ValueError(f"No rows found at {gs_path}")
        samples = [Sample.from_dict(r) for r in rows]
        return samples

    @property
    def last_samples(self) -> list[Sample]:
        """Return examples aligned with last batch."""
        return self._last_samples

    def __call__(self, num_prompts: int) -> list[Sample]:
        start_idx = self._current_idx
        end_idx = self._current_idx + num_prompts
        total = len(self.samples)
        indices = [i % total for i in range(start_idx, end_idx)]
        samples = [self.samples[i] for i in indices]
        self._last_samples = samples
        self._current_idx = end_idx % total
        return samples

    def prepare_batch(self, samples: list[Sample], generations: InferenceResults) -> RLBatch:

        tokenizer = AutoTokenizer.from_pretrained(self.hf_model)

        tokens = jnp.array(
            [
                jnp.stack(
                    [
                        jnp.pad(
                            jnp.array(tokens),
                            (self.seq_length - tokens.shape[0], 0),
                            mode="constant",
                            constant_values=0,
                        )
                        for tokens in inference_rollout.rollouts
                    ]
                )
                for inference_rollout in generations.rollouts
            ],
            dtype=jnp.int32,
        )

        reference_model_logprobs = jnp.array(
            [
                jnp.stack(
                    [
                        jnp.pad(
                            jnp.array(logprobs),
                            (self.seq_length - logprobs.shape[0], 0),
                            mode="constant",
                            constant_values=-jnp.inf,
                        )
                        for logprobs in inference_rollout.logprobs
                    ]
                )
                for inference_rollout in generations.rollouts
            ],
            dtype=jnp.float32,
        )

        seq_lens = jnp.array(
            [[len(tokens) for tokens in inference_rollout.rollouts] for inference_rollout in generations.rollouts],
            dtype=jnp.int32,
        )

        rewards = jnp.array(
            [
                [self.get_reward(tokenizer.decode(tokens), sample.answer) for tokens in inference_rollout.rollouts]
                for sample, inference_rollout in zip(samples, generations.rollouts)
            ],
            dtype=jnp.float32,
        )

        token_mask = jnp.where(jnp.isfinite(reference_model_logprobs), 1, 0).astype(jnp.int32)

        return RLBatch(tokens, reference_model_logprobs, seq_lens, rewards, token_mask)

    def get_reward(self, output_str: str, answer: str) -> float:
        return self.verifier(VerifierInput(output_str, answer))

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
