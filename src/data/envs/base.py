import abc
import hashlib
import random
from collections.abc import Callable, Mapping
from functools import cached_property
from typing import Any, ClassVar

from hydra.utils import instantiate
from stax.logger import staxLogger as logger

from ..config import DatasetConfig, Score

SAMPLE_SEED = 0


class Loader(staticmethod):
    dataset_name: str


def register_dataset(name: str) -> Callable[[Callable], Loader]:
    def decorator(loader: Callable) -> Loader:
        registered = Loader(loader)
        registered.dataset_name = name
        return registered

    return decorator


class Environment[ProblemT](abc.ABC):
    name: ClassVar[str]
    system_prompt: str | None = None
    registry: ClassVar[dict[str, Callable[[], list]]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.registry = {v.dataset_name: v.__func__ for v in vars(cls).values() if isinstance(v, Loader)}

    def __init__(
        self,
        datasets: Mapping[str, Mapping[str, Any] | None],
        use_system_prompt: bool = False,
    ) -> None:
        self.datasets = {name: DatasetConfig(**(config or {})) for name, config in datasets.items()}
        self.system_prompt = self.system_prompt if use_system_prompt else None
        self.problems: list[ProblemT] = []
        self.dataset_names: list[str] = []

    def load(self) -> int:
        self.problems, self.dataset_names = [], []
        for name, dataset in self.datasets.items():
            problems = self.subsample(self.registry[name](), dataset.n_samples)
            logger.info(f"[env] {self.name}/{name}: {len(problems)} problems")
            self.problems += problems
            self.dataset_names += [name] * len(problems)
        return len(self.problems)

    def subsample(self, problems: list[ProblemT], n_samples: int | None) -> list[ProblemT]:
        if n_samples is None or n_samples >= len(problems):
            return problems
        ids = sorted(random.Random(SAMPLE_SEED).sample(range(len(problems)), n_samples))
        return [problems[i] for i in ids]

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "problems": [], "dataset_names": []}

    @cached_property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for problem in self.problems:
            digest.update(self.question(problem).encode())
            digest.update(b"\0")
        return digest.hexdigest()

    def render(self, id: int) -> tuple[str | None, str]:
        return self.system_prompt, self.user_prompt(self.question(self.problems[id]))

    @abc.abstractmethod
    def question(self, problem: ProblemT) -> str: ...

    @abc.abstractmethod
    def user_prompt(self, question: str) -> str: ...

    @abc.abstractmethod
    def score(self, problem: ProblemT, completion: str) -> Score: ...


def load_envs(configs: Mapping[str, Any]) -> dict[str, Environment]:
    envs = {name: instantiate(config) for name, config in configs.items()}
    for name, env in envs.items():
        if name != env.name:
            raise ValueError(f"Config key {name} does not match environment {env.name}")
        env.load()
    return envs
