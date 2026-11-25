import zlib
from collections import deque
from queue import Empty, Queue
from threading import Lock
from typing import Any

import numpy as np

from .config import DataEvent, ReplayKey, Sample, SampleKey
from .envs import Environment


class EnvStream:
    def __init__(self, env: Environment, seed: int, epoch: int = 0, idx: int = 0) -> None:
        if not env.problems:
            raise ValueError(f"Environment {env.name} has no problems")
        self.env = env
        self.seed = seed
        self.epoch = epoch
        self.idx = idx
        self.order = self.shuffle()

    def advance(self) -> SampleKey:
        if self.idx == len(self.order):
            self.epoch += 1
            self.idx = 0
            self.order = self.shuffle()
        key = SampleKey(self.env.name, self.epoch, self.order[self.idx].item())
        self.idx += 1
        return key

    def shuffle(self) -> np.ndarray:
        name = zlib.crc32(self.env.name.encode())
        ids = np.arange(len(self.env.problems), dtype=np.int32)
        return np.random.default_rng([self.seed, name, self.epoch]).permutation(ids)

    def sample(self, key: SampleKey, n_rollouts: int) -> Sample:
        return Sample(*key, *self.env.render(key.id), n_rollouts=n_rollouts)

    def save(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "idx": self.idx, "hash": self.env.fingerprint}

    def restore(self, state: dict[str, Any]) -> None:
        if state["hash"] != self.env.fingerprint:
            raise ValueError(f"Environment {self.env.name} changed since the checkpoint")
        self.epoch = state["epoch"]
        self.idx = state["idx"]
        self.order = self.shuffle()


class DataLoader:
    def __init__(
        self, envs: dict[str, Environment], events: Queue, seed: int, n_rollouts: int, max_retries: int = 2
    ) -> None:
        self.events = events
        self.seed = seed
        self.n_rollouts = n_rollouts
        self.max_retries = max_retries
        self.streams = {name: EnvStream(env, seed) for name, env in envs.items()}
        self.names = list(envs)
        self.n_samples_per_env = [len(env.problems) for env in envs.values()]
        self.draw = 0
        self.replay: deque[ReplayKey] = deque()
        self.pending: dict[SampleKey, int] = {}
        self.finished: dict[SampleKey, int] = {}
        self.lock = Lock()

    def next(self) -> Sample:
        with self.lock:
            self.drain()
            key, retries = self.sample()
            self.pending[key] = retries
        return self.streams[key.env].sample(key, self.n_rollouts)

    def sample(self) -> ReplayKey:
        if self.replay:
            return self.replay.popleft()
        return ReplayKey(self.sample_env().advance(), 0)

    def sample_env(self) -> EnvStream:
        counts = np.asarray(self.n_samples_per_env, dtype=np.float64)
        index = np.random.default_rng([self.seed, self.draw]).choice(len(counts), p=counts / counts.sum())
        self.draw += 1
        return self.streams[self.names[index]]

    def drain(self) -> None:
        while True:
            try:
                event, key = self.events.get_nowait()
            except Empty:
                return
            self.apply(DataEvent(event), SampleKey(*key))

    def apply(self, event: DataEvent, key: SampleKey) -> None:
        if event == DataEvent.FINISHED:
            self.finished[key] = self.pending.pop(key)
        elif event == DataEvent.CONSUMED:
            del self.finished[key]
        elif event == DataEvent.DROPPED:
            source = self.finished if key in self.finished else self.pending
            retries = source.pop(key)
            if retries < self.max_retries:
                self.replay.appendleft(ReplayKey(key, retries + 1))

    def save(self) -> dict[str, Any]:
        with self.lock:
            self.drain()
            return {
                "draw": self.draw,
                "envs": {name: stream.save() for name, stream in self.streams.items()},
                "replay": list(self.replay),
                "finished": list(self.finished.items()),
                "pending": list(self.pending.items()),
            }

    def restore(self, state: dict[str, Any]) -> None:
        if set(state["envs"]) != set(self.streams):
            raise ValueError(f"Checkpoint has envs {sorted(state['envs'])}, but config has {sorted(self.streams)}")
        for name, saved in state["envs"].items():
            self.streams[name].restore(saved)
        self.draw = state["draw"]
        unfinished = state["replay"] + state["finished"] + state["pending"]
        self.replay = deque(ReplayKey(SampleKey(*key), retries) for key, retries in unfinished)
        self.pending, self.finished = {}, {}
