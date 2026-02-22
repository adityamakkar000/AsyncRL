from typing import Any

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

from src.constants import DATA, GS_BUCKET
from src.data.config import DatasetConfig, RLBatch, Sample
from src.data.utils import compute_aux_metrics, load_jsonl_from_gcs
from src.data.verifier import Verifier, VerifierInput
from src.inference_engine.config import InferenceResults, InferenceRollout


class DataLoader:
    def __init__(
        self,
        dataset_config: DatasetConfig,
        max_seq_length: int,
        hf_model: str,
    ) -> None:
        self.dataset_config = dataset_config
        self.max_seq_length = max_seq_length
        self.samples = self._load_from_gcs()
        self._last_samples = []
        self._current_idx = 0
        self.verifier = Verifier()
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)

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

    def _get_rewards(self, samples: list[Sample], generations: InferenceResults) -> tuple[jax.Array, int]:
        num_unparsable = 0
        total_rewards = []
        for sample, inference_rollout in zip(samples, generations.rollouts):
            token_rewards = []
            for tokens in inference_rollout.rollouts:
                reward = self.get_reward(self.tokenizer.decode(tokens), sample.answer)
                if reward is None:
                    num_unparsable += 1
                    reward = 0.0
                token_rewards.append(reward)

            total_rewards.append(token_rewards)

        return jnp.array(total_rewards, dtype=jnp.int32), num_unparsable

    def prepare_batch(self, samples: list[Sample], generations: InferenceResults) -> tuple[RLBatch, int]:
        tokens = self.pad_tokens(generations.rollouts, self.tokenizer.pad_token_id, "rollouts")
        reference_model_logprobs = self.pad_tokens(generations.rollouts, -jnp.inf, "logprobs")

        seq_lens = jnp.array(
            [[len(tokens) for tokens in inference_rollout.rollouts] for inference_rollout in generations.rollouts],
            dtype=jnp.int32,
        )

        rewards, num_unparsable = self._get_rewards(samples, generations)

        token_mask = reference_model_logprobs != -jnp.inf
        group_mean = rewards.mean(axis=1, keepdims=True) * jnp.ones_like(rewards)
        group_std = rewards.std(axis=1, keepdims=True) * jnp.ones_like(rewards) + 1e-8

        rl_batch = RLBatch(tokens, reference_model_logprobs, seq_lens, rewards, group_mean, group_std, token_mask)

        def compress(x):
            x = x.reshape(x.shape[0] * x.shape[1], -1)
            return x.squeeze(-1) if x.shape[-1] == 1 else x

        rl_batch = jax.tree.map(compress, rl_batch)

        return rl_batch, num_unparsable

    def pad_tokens(self, inference_rollouts: list[InferenceRollout], constant_val, field_name: str) -> jax.Array:
        for inference_rollout in inference_rollouts:
            for field in getattr(inference_rollout, field_name):
                assert self.max_seq_length >= field.shape[0], (
                    f"self.max_seq_length ({self.max_seq_length}) must be >= field length ({field.shape[0]})"
                )

        return jnp.array(
            [
                jnp.stack(
                    [
                        jnp.pad(
                            jnp.array(field),
                            (self.max_seq_length - field.shape[0], 0),
                            mode="constant",
                            constant_values=constant_val,
                        )
                        for field in getattr(inference_rollout, field_name)
                    ]
                )
                for inference_rollout in inference_rollouts
            ]
        )

    def get_reward(self, output_str: str, answer: str) -> float | None:
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
