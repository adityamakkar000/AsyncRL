import jax
import numpy as np

from .config import InferenceRollout, RLBatch, ScoredRollout
from .envs import Environment


def score_rollout(env: Environment, rollout: InferenceRollout) -> ScoredRollout:
    return ScoredRollout(rollout, [env.grade(rollout.sample.id, text) for text in rollout.rollout_strs])


def compute_aux_metrics(batch: RLBatch) -> dict[str, float]:
    token_mask = batch.token_mask
    return {
        "mean_reward": np.mean(batch.rewards).item(),
        "std_reward": np.std(batch.rewards).item(),
        "max_reward": np.max(batch.rewards).item(),
        "min_reward": np.min(batch.rewards).item(),
        "mean_length": np.mean(token_mask.sum(axis=1)).item(),
        "median_length": np.median(token_mask.sum(axis=1)).item(),
    }


def pad(groups: list[list[np.ndarray]], value: float, max_seq_len: int) -> np.ndarray:
    return np.array([[np.pad(x, (0, max_seq_len - len(x)), constant_values=value) for x in group] for group in groups])


def flatten_groups(x: np.ndarray) -> np.ndarray:
    x = x.reshape(x.shape[0] * x.shape[1], -1)
    return x.squeeze(-1) if x.shape[-1] == 1 else x


def collate(scored: list[ScoredRollout], pad_token: int, max_seq_len: int) -> tuple[RLBatch, dict[str, float]]:
    rollouts = [s.rollout for s in scored]
    tokens = pad([r.rollout_tokens for r in rollouts], pad_token, max_seq_len)
    logprobs = pad([r.rollout_logprobs for r in rollouts], -np.inf, max_seq_len)
    seq_lens = np.array([[len(t) for t in r.rollout_tokens] for r in rollouts], dtype=np.int32)
    rewards = np.stack([s.rewards for s in scored])

    batch = jax.tree.map(
        flatten_groups,
        RLBatch.from_numpy(
            tokens,
            np.where(np.isfinite(logprobs), logprobs, 0.0),
            seq_lens,
            rewards,
            np.broadcast_to(rewards.mean(axis=1, keepdims=True), rewards.shape),
            np.broadcast_to(rewards.std(axis=1, keepdims=True) + 1e-8, rewards.shape),
            token_mask=logprobs != -np.inf,
        ),
    )

    num_unparsable = sum(score.status == "unparsable" for s in scored for score in s.scores)
    metrics = {"num_unparsable": num_unparsable / rewards.size} | compute_aux_metrics(batch)
    return batch, {f"train/{k}": v for k, v in metrics.items()}


def train_table(scored: list[ScoredRollout]) -> dict[str, list[str]]:
    rows = [
        (s.rollout.sample.env, text, score) for s in scored for text, score in zip(s.rollout.rollout_strs, s.scores)
    ]
    return {
        "env": [env for env, _, _ in rows],
        "rollout": [text for _, text, _ in rows],
        "expected": [str(score.expected) for _, _, score in rows],
        "extracted": [str(score.extracted) for _, _, score in rows],
        "reward": [str(score.reward) for _, _, score in rows],
    }
