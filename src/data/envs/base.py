import abc
import hashlib
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property
from typing import Any, ClassVar

from hydra.utils import instantiate
from stax.logger import staxLogger as logger

from ..config import DatasetConfig, Score

Filter = Callable[[Any, "Environment"], bool]
Transform = Callable[[list, Callable[[Any], str]], list]


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
        datasets: Mapping[str, DatasetConfig],
        filters: Sequence[Filter] = (),
        transforms: Sequence[Transform] = (),
        use_system_prompt: bool = False,
    ) -> None:
        self.datasets = datasets
        self.filters = filters
        self.transforms = transforms
        self.system_prompt = self.system_prompt if use_system_prompt else None
        self.problems: list[ProblemT] = []
        self.dataset_names: list[str] = []

    def load(self) -> int:
        self.problems, self.dataset_names = [], []
        for name, dataset in self.datasets.items():
            problems = self.process(name, dataset, self.registry[name]())
            self.problems += problems
            self.dataset_names += [name] * len(problems)
        return len(self.problems)

    def process(self, name: str, dataset: DatasetConfig, problems: list[ProblemT]) -> list[ProblemT]:
        kept = problems
        for keep in [*self.filters, *dataset.filters]:
            kept = [problem for problem in kept if keep(problem, self)]
        for transform in [*self.transforms, *dataset.transforms]:
            kept = transform(kept, self.question)
        logger.info(f"[env] {self.name}/{name}: kept {len(kept)}/{len(problems)}")
        return kept

    @cached_property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for problem in self.problems:
            digest.update(self.question(problem).encode())
            digest.update(b"\0")
        return digest.hexdigest()

    def render(self, id: int) -> tuple[str | None, str]:
        return self.system_prompt, self.user_prompt(self.question(self.problems[id]))

    def grade(self, id: int, completion: str) -> Score:
        return self.score(self.problems[id], completion)

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
