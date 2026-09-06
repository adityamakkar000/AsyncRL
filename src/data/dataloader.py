import random
from typing import Any

import jax
import numpy as np
from hydra.utils import instantiate
from stax.logger import staxLogger as logger

from .config import DatasetConfig, InferenceRollout, RLBatch, Sample
from .filters import Filter
from .register import GLOBAL_DICT
from .transforms import Transform
from .utils import compute_aux_metrics, decode_tokens, load_tokenizer, pass_at_k, resolve_pad_eos
from .verifier import Verifier


class DataLoader:
    def __init__(
        self,
        dataset_config: DatasetConfig,
        max_seq_length: int,
        use_system_prompt: bool,
        hf_model: str,
    ) -> None:
        self.dataset_config = dataset_config
        self.max_seq_length = max_seq_length
        self.use_system_prompt = use_system_prompt
        self.tokenizer = load_tokenizer(hf_model)
        self.pad_token, self.eos_token = resolve_pad_eos(self.tokenizer)

        self.verifier: Verifier = instantiate(dataset_config.verifier)
        self.transforms: list[Transform] = [instantiate(t) for t in dataset_config.transforms]
        self.filters: list[Filter] = [instantiate(f) for f in dataset_config.filters]
        for data_filter in self.filters:
            data_filter.bind(self.tokenizer, self.use_system_prompt)

        self._current_idx = 0
        self.samples = self.load_samples()
        self.total_samples = len(self.samples)

    def load_samples(self) -> list[Sample]:
        name = self.dataset_config.name
        if name not in GLOBAL_DICT:
            raise ValueError(f"Unknown dataset {name}. Registered datasets: {sorted(GLOBAL_DICT)}")

        samples = GLOBAL_DICT[name]()
        if not samples:
            raise ValueError(f"Dataset {name} returned no samples")
        return self.process_samples(samples)

    def process_samples(self, samples: list[Sample]) -> list[Sample]:
        kept = samples
        for data_filter in self.filters:
            before = len(kept)
            kept = data_filter.select(kept)
            logger.info(
                f"[dataset] {self.dataset_config.name}: {type(data_filter).__name__} dropped {before - len(kept)}"
            )
        for transform in self.transforms:
            kept, n = transform(kept)
            logger.info(f"[dataset] {self.dataset_config.name}: {type(transform).__name__} reported {n}")
        logger.info(
            f"[dataset] {self.dataset_config.name}: kept {len(kept)}/{len(samples)} samples "
            f"after {len(self.filters)} filter(s) and {len(self.transforms)} transform(s)"
        )
        return kept

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
                reward = self.get_reward(decode_tokens(self.tokenizer, tokens), inference_rollout.sample.answer)
                if reward is None:
                    num_unparsable += 1
                    reward = 0.0
                token_rewards.append(reward)

            total_rewards.append(token_rewards)

        return np.array(total_rewards, dtype=np.float32), num_unparsable

    def prepare_batch(self, generations: list[InferenceRollout], train: bool) -> tuple[RLBatch, dict]:
        tokens = self.pad_tokens(generations, self.pad_token, "rollout_tokens")
        reference_model_logprobs = self.pad_tokens(generations, -np.inf, "rollout_logprobs")

        seq_lens = np.array(
            [[len(tokens) for tokens in inference_rollout.rollout_tokens] for inference_rollout in generations],
            dtype=np.int32,
        )

        rewards, num_unparsable = self._get_rewards(generations)

        group_mean = rewards.mean(axis=1, keepdims=True) * np.ones_like(rewards, dtype=np.float32)
        group_std = rewards.std(axis=1, keepdims=True) * np.ones_like(rewards, dtype=np.float32) + 1e-8

        def compress(x):
            x = x.reshape(x.shape[0] * x.shape[1], -1)
            return x.squeeze(-1) if x.shape[-1] == 1 else x

        rl_batch = jax.tree.map(
            compress,
            RLBatch.from_numpy(
                tokens,
                np.where(np.isfinite(reference_model_logprobs), reference_model_logprobs, 0.0),
                seq_lens,
                rewards,
                group_mean,
                group_std,
                token_mask=(reference_model_logprobs != -np.inf),
            ),
        )

        num_unparsable = num_unparsable / rewards.size

        prefix = "train" if train else "val"
        metrics = {"num_unparsable": num_unparsable} | compute_aux_metrics(rl_batch)
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        return rl_batch, metrics

    def score_rollouts(
        self, generations: list[InferenceRollout], k: int, pass_k: list[int]
    ) -> tuple[dict[str, float], np.ndarray]:
        rewards, num_unparsable = self._get_rewards(generations)
        iterations = [i for g in generations for i in g.weight_iteration]
        n_correct = (rewards > 0).sum(axis=1)

        metrics = {
            f"avg@{k}": rewards.mean().item(),
            "num_unparsable": num_unparsable / rewards.size,
            "n_prompts": float(len(generations)),
            "weight_iteration_min": float(min(iterations)) if iterations else 0.0,
            "weight_iteration_max": float(max(iterations)) if iterations else 0.0,
        }
        for target_k in pass_k:
            label = f"pass@{target_k}" if target_k <= k else f"pass@k={target_k}"
            metrics[label] = float(np.mean([pass_at_k(k, int(c), target_k) for c in n_correct]))
        return metrics, rewards

    def build_trace_table(
        self, generations: list[InferenceRollout], rewards: np.ndarray, n_prompts: int
    ) -> dict[str, list[str]]:
        table: dict[str, list[str]] = {"prompt": [], "answer": [], "rollout": [], "extracted": [], "reward": []}
        indices = range(len(generations))
        if 0 <= n_prompts < len(generations):
            indices = sorted(random.Random(0).sample(indices, n_prompts))

        for prompt_idx in indices:
            rollout = generations[prompt_idx]
            for rollout_idx, text in enumerate(rollout.rollout_strs):
                table["prompt"].append(rollout.sample.prompt)
                table["answer"].append(rollout.sample.answer)
                table["rollout"].append(text)
                table["extracted"].append(str(self.verifier.extract(text)))
                table["reward"].append(str(rewards[prompt_idx][rollout_idx]))
        return table

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

    def check_rollout_zero_variance(self, InferenceRollout: InferenceRollout) -> bool:
        reward_set = set()
        for tokens in InferenceRollout.rollout_tokens:
            reward = self.get_reward(decode_tokens(self.tokenizer, tokens), InferenceRollout.sample.answer)
            if reward is None:
                reward = 0.0
            reward_set.add(reward)
            if len(reward_set) > 1:
                return False

        return True

    def get_reward(self, output_str: str, answer: str) -> float | None:
        return self.verifier.get_reward(output_str, answer)

    def save_checkpoint(self) -> dict[str, Any]:
        return {
            "current_idx": self._current_idx,
        }

    def restore_checkpoint(self, state: dict[str, Any]) -> None:
        if "current_idx" not in state:
            raise ValueError("Missing 'current_idx' in checkpoint state")
        self._current_idx = state["current_idx"]
