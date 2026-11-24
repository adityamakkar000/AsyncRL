import re
from typing import Any

from ...envs import Environment

CHINESE_CHARACTERS = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


class EnglishFilter:
    def __call__(self, problem: Any, env: Environment) -> bool:
        return not any(isinstance(value, str) and CHINESE_CHARACTERS.search(value) for value in vars(problem).values())
