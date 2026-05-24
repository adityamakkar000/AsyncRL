from typing import Any

import jax
import numpy as np
from transformers import AutoTokenizer

from src.constants import DATA, GS_BUCKET

from .config import DatasetConfig, InferenceRollout, RLBatch, Sample
from .utils import compute_aux_metrics, load_jsonl_from_gcs
from .verifier import Verifier


class DataLoader:
    def __init__(
        self, dataset_config: DatasetConfig, max_seq_length: int, hf_model: str, mesh: jax.sharding.Mesh
    ) -> None:
        self.dataset_config = dataset_config
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
        self.mesh = mesh

        self.samples = self._load_from_gcs()
        self.total_samples = len(self.samples)

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
        self._current_idx = 0
        return samples

    def __call__(self, num_prompts: int) -> list[Sample]:
        samples = [
            self.samples[i % self.total_samples] for i in range(self._current_idx, self._current_idx + num_prompts)
        ]
        self._current_idx = (self._current_idx + num_prompts) % self.total_samples
        return samples

    def _get_rewards(self, generations: list[InferenceRollout]) -> tuple[np.ndarray, int]:
        num_unparsable = 0
        total_rewards = []
        for inference_rollout in generations:
            token_rewards = []
            for tokens in inference_rollout.rollout_tokens:
                reward = self.get_reward(self.tokenizer.decode(tokens), inference_rollout.sample.answer)
                if reward is None:
                    num_unparsable += 1
                    reward = 0.0
                token_rewards.append(reward)

            total_rewards.append(token_rewards)

        return np.array(total_rewards, dtype=np.int32), num_unparsable

    def prepare_batch(self, generations: list[InferenceRollout], train: bool) -> tuple[RLBatch, dict]:
        tokens = self.pad_tokens(generations, self.tokenizer.pad_token_id, "rollout_tokens")
        reference_model_logprobs = self.pad_tokens(generations, -np.inf, "rollout_logprobs")

        seq_lens = np.array(
            [[len(tokens) for tokens in inference_rollout.rollout_tokens] for inference_rollout in generations],
            dtype=np.int32,
        )

        rewards, num_unparsable = self._get_rewards(generations)

        group_mean = rewards.mean(axis=1, keepdims=True) * np.ones_like(rewards)
        group_std = rewards.std(axis=1, keepdims=True) * np.ones_like(rewards) + 1e-8

        def compress(x):
            x = x.reshape(x.shape[0] * x.shape[1], -1)
            return x.squeeze(-1) if x.shape[-1] == 1 else x

        rl_batch = jax.tree.map(
            compress, RLBatch(tokens, reference_model_logprobs, seq_lens, rewards, group_mean, group_std)
        )

        num_unparsable = num_unparsable / self.dataset_config.batch_size

        prefix = "train" if train else "val"
        metrics = {"num_unparsable": num_unparsable} | compute_aux_metrics(rl_batch)
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        return rl_batch, metrics

    def pad_tokens(self, inference_rollouts: list[InferenceRollout], constant_val, field_name: str) -> np.ndarray:
        for inference_rollout in inference_rollouts:
            for field in getattr(inference_rollout, field_name):
                assert self.max_seq_length >= field.shape[0], (
                    f"self.max_seq_length ({self.max_seq_length}) must be >= field length ({field.shape[0]})"
                )

        return np.array(
            [
                np.stack(
                    [
                        np.pad(
                            np.array(field),
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
        return Verifier.get_reward(output_str, answer)

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
