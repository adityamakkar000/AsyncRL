import os
import threading
import traceback
from collections import Counter
from collections.abc import Callable
from typing import Any

from stax.logger import staxLogger as logger

from .batch import collate, train_table
from .config import DataEvent, InferenceRollout, RLBatch, ScoredRollout, ScoreStatus
from .dataloader import DataLoader
from .envs import load_envs
from .scorer import Scorer


class DataWorker:
    def __init__(
        self,
        env_configs: dict[str, Any],
        queues: Any,
        rank: int,
        seed: int,
        n_prompts: int,
        group_size: int,
        pad_token: int,
        max_seq_len: int,
        max_lag: int,
        filter_zero_variance: bool,
        timeout: int,
    ) -> None:
        self.envs = load_envs(env_configs)
        n_processes = max(1, (os.cpu_count() or 2) // 2)
        self.scorer = Scorer(n_processes, max_pending=2 * n_processes)
        self.queues = queues
        self.n_prompts = n_prompts
        self.pad_token = pad_token
        self.max_seq_len = max_seq_len
        self.max_lag = max_lag
        self.filter_zero_variance = filter_zero_variance
        self.timeout = timeout
        self.loader = DataLoader(self.envs, queues.data_events_queue, seed, group_size) if rank == 0 else None
        self.weight_iteration = 0
        self.filtered: Counter[str] = Counter()
        self.filtered_lock = threading.Lock()
        self.scored: list[ScoredRollout] = []
        self.stopped = threading.Event()

    def start(self) -> None:
        if self.loader is not None:
            self.spawn(self.fill_next)
        self.spawn(self.submit_next)
        self.spawn(self.add_next)

    def next(self, weight_iteration: int) -> tuple[RLBatch, dict[str, float], dict[str, float]]:
        self.weight_iteration = weight_iteration
        self.scored = []
        while len(self.scored) < self.n_prompts:
            item: ScoredRollout = self.queues.scored_queue.get(timeout=self.timeout)
            if not self.drop_if_stale(item.rollout):
                self.queues.data_events_queue.put((DataEvent.CONSUMED, item.rollout.sample.key))
                self.scored.append(item)

        batch, batch_metrics = collate(self.scored, self.pad_token, self.max_seq_len)
        rollout_metrics = self.rollout_metrics()
        return batch, rollout_metrics, batch_metrics

    def table(self) -> dict[str, list[str]]:
        return train_table(self.scored)

    def save(self) -> dict[str, Any]:
        assert self.loader is not None, "Only the dataloader host can save"
        return self.loader.save()

    def restore(self, state: dict[str, Any]) -> None:
        assert self.loader is not None, "Only the dataloader host can restore"
        self.loader.restore(state)

    def cleanup(self) -> None:
        self.stopped.set()
        self.scorer.close()

    def spawn(self, step: Callable[[], None]) -> None:
        def run_forever():
            try:
                while not self.stopped.is_set():
                    step()
            except Exception:  # noqa: BLE001
                if self.stopped.is_set():
                    return
                traceback.print_exc()
                os._exit(1)

        threading.Thread(target=run_forever, daemon=True).start()

    def fill_next(self) -> None:
        assert self.loader is not None
        self.queues.prompt_queue.put(self.loader.next())

    def submit_next(self) -> None:
        rollout: InferenceRollout = self.queues.rollout_queue.get()
        if not self.drop_if_stale(rollout):
            self.scorer.submit(self.envs[rollout.sample.env], rollout)

    def add_next(self) -> None:
        scored = self.scorer.get()
        if self.filter_zero_variance and scored.zero_variance:
            self.drop(scored.rollout, "zero_variance")
            return

        self.queues.data_events_queue.put((DataEvent.FINISHED, scored.rollout.sample.key))
        self.queues.scored_queue.put(scored)

    def drop_if_stale(self, rollout: InferenceRollout) -> bool:
        stale = self.lag(rollout) > self.max_lag
        if stale:
            self.drop(rollout, "lag")
        return stale

    def drop(self, rollout: InferenceRollout, reason: str) -> None:
        self.queues.data_events_queue.put((DataEvent.DROPPED, rollout.sample.key))
        with self.filtered_lock:
            self.filtered[reason] += 1

    def lag(self, rollout: InferenceRollout) -> int:
        return self.weight_iteration - rollout.lag

    def rollout_metrics(self) -> dict[str, float]:
        lags = [self.lag(item.rollout) for item in self.scored]
        with self.filtered_lock:
            filtered, self.filtered = self.filtered, Counter()

        n_filtered = filtered.total()
        if n_filtered >= 10:
            logger.warning(
                f"Filtered {n_filtered} rollouts due to lag or zero variance, consider increasing max_lag or disabling zero variance filtering if this is happening frequently.",
                log_for_all=True,
            )

        scores = [score for item in self.scored for score in item.scores]
        return {
            "train/num_judged": sum(score.status == ScoreStatus.JUDGED for score in scores) / len(scores),
            "train/max_off_policy": max(lags),
            "train/min_off_policy": min(lags),
            "train/mean_off_policy": sum(lags) / len(lags),
            "train/lag_filter": filtered["lag"],
            "train/zero_variance_filter": filtered["zero_variance"],
        }
