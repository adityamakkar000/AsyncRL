import os
import random
from collections import defaultdict
from typing import Any

import numpy as np
from stax.logger import staxLogger as logger
from stax.writer import TableMetrics
from tqdm import tqdm

from .config import Sample, ScoredRollout, ScoreStatus, Split
from .envs import load_envs
from .scorer import Scorer

EVAL_PROCESSES = max(1, (os.cpu_count() or 2) // 4)


def pass_at_k(n: int, c: int, k: int) -> float:
    if k > n:
        return float(1.0 - (1.0 - c / n) ** k)
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def eval_metrics(scored: list[ScoredRollout], group_size: int, pass_k: list[int]) -> dict[str, float]:
    rewards = np.stack([s.rewards for s in scored])
    iterations = [i for s in scored for i in s.rollout.weight_iteration]
    n_correct = (rewards > 0).sum(axis=1)
    n_unparsable = sum(score.status == ScoreStatus.UNPARSABLE for s in scored for score in s.scores)
    n_judged = sum(score.status == ScoreStatus.JUDGED for s in scored for score in s.scores)

    metrics = {
        f"avg@{group_size}": rewards.mean().item(),
        "num_unparsable": n_unparsable / rewards.size,
        "num_judged": n_judged / rewards.size,
        "n_prompts": float(len(scored)),
        "weight_iteration_min": float(min(iterations)) if iterations else 0.0,
        "weight_iteration_max": float(max(iterations)) if iterations else 0.0,
    }
    for k in pass_k:
        label = f"pass@{k}" if k <= group_size else f"pass@k={k}"
        metrics[label] = float(np.mean([pass_at_k(group_size, int(c), k) for c in n_correct]))
    return metrics


def trace_table(scored: list[ScoredRollout], n_prompts: int) -> dict[str, list[str]]:
    if 0 <= n_prompts < len(scored):
        scored = [scored[i] for i in sorted(random.Random(0).sample(range(len(scored)), n_prompts))]

    rows = [
        (s.rollout.sample.prompt, text, score) for s in scored for text, score in zip(s.rollout.rollout_strs, s.scores)
    ]
    return {
        "prompt": [prompt for prompt, _, _ in rows],
        "rollout": [text for _, text, _ in rows],
        "expected": [str(score.expected) for _, _, score in rows],
        "extracted": [str(score.extracted) for _, _, score in rows],
        "reward": [str(score.reward) for _, _, score in rows],
    }


class EvalWorker:
    def __init__(self, config: Any, queues: Any, n_train_workers: int, rank: int, timeout: int) -> None:
        self.config = config
        self.envs = load_envs(config.eval_envs)
        self.scorer = Scorer(EVAL_PROCESSES) if rank == 0 else None
        self.queues = queues
        self.n_train_workers = n_train_workers
        self.rank = rank
        self.timeout = timeout

    def run_eval(self) -> tuple[dict[str, float], list[TableMetrics]]:
        if self.rank != 0:
            self.queues.eval_done_queue.get(timeout=self.timeout)
            return {}, []

        metrics: dict[str, float] = {}
        tables: list[TableMetrics] = []
        for name, scored in self.group(self.collect(self.submit())).items():
            scores = eval_metrics(scored, self.config.group_size, list(self.config.pass_k))
            metrics |= {f"eval/{name}/{key}": value for key, value in scores.items()}
            if self.config.log_traces_n_prompts != 0:
                traces = trace_table(scored, self.config.log_traces_n_prompts)
                tables.append(TableMetrics(f"eval/generations/{name}", traces))

        for _ in range(self.n_train_workers - 1):
            self.queues.eval_done_queue.put("done")
        return metrics, tables

    def submit(self) -> int:
        samples = [
            Sample(e.name, 0, id, *e.render(id), n_rollouts=self.config.group_size, split=Split.EVAL)
            for e in self.envs.values()
            for id in range(len(e.problems))
        ]
        for sample in samples:
            self.queues.eval_prompt_queue.put(sample)
        logger.info(f"[eval] waiting on {len(samples)} prompts", log_for_all=True)
        return len(samples)

    def collect(self, n_prompts: int) -> list[ScoredRollout]:
        assert self.scorer is not None, "Only rank 0 scores evals"
        with tqdm(total=n_prompts, desc="[eval] rollouts", unit="prompt", mininterval=10) as bar:
            for _ in range(n_prompts):
                rollout = self.queues.eval_rollout_queue.get(timeout=self.timeout)
                self.scorer.submit(self.envs[rollout.sample.env], rollout)
                bar.update(1)
        return [self.scorer.get() for _ in range(n_prompts)]

    def group(self, scored_rollouts: list[ScoredRollout]) -> dict[str, list[ScoredRollout]]:
        groups: dict[str, list[ScoredRollout]] = defaultdict(list)
        for scored in scored_rollouts:
            env = self.envs[scored.rollout.sample.env]
            groups[f"{env.name}/{env.dataset_names[scored.rollout.sample.id]}"].append(scored)
        return groups

    def cleanup(self) -> None:
        if self.scorer is not None:
            self.scorer.close()
