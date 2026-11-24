import re
from typing import Any

from ...envs import Environment

FIGURE_WORDS = re.compile(r"figure|diagram|shown below", re.IGNORECASE)


class FigureFilter:
    def __call__(self, problem: Any, env: Environment) -> bool:
        return not FIGURE_WORDS.search(env.question(problem))
