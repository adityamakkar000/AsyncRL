import multiprocessing as mp
import os
import queue
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any

from .config import InferenceRollout, Score, ScoredRollout
from .envs import Environment

SLEEP = 5


def exit_with_parent(parent: int) -> None:
    def watch() -> None:
        while os.getppid() == parent:
            time.sleep(SLEEP)
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


def grade(env: Environment, problem: Any, completions: list[str]) -> list[Score]:
    return [env.score(problem, completion) for completion in completions]


class Scorer:
    def __init__(self, n_processes: int, max_pending: int = 0) -> None:
        self.n_processes = n_processes
        self.executor = ProcessPoolExecutor(
            n_processes, mp_context=mp.get_context("spawn"), initializer=exit_with_parent, initargs=(os.getpid(),)
        )
        self.pending: queue.Queue[tuple[InferenceRollout, Future[list[Score]]]] = queue.Queue(max_pending)

    def submit(self, env: Environment, rollout: InferenceRollout) -> None:
        future = self.executor.submit(grade, env, env.problems[rollout.sample.id], rollout.rollout_strs)
        self.pending.put((rollout, future))

    def get(self) -> ScoredRollout:
        rollout, future = self.pending.get()
        return ScoredRollout(rollout, future.result())

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
