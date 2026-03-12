import re
from dataclasses import dataclass
from loguru import logger


@dataclass
class VerifierInput:
    solution: str
    answer: str


class Verifier:
    def __init__(self):
        self.box_pattern = re.compile(r"\\boxed\{")

    def _remove_think(self, solution: str) -> str:
        end_tag = "</think>"
        idx = solution.rfind(end_tag)
        if idx == -1:
            return solution
        return solution[idx + len(end_tag) :]

    def extract_boxed_content(self, solution: str) -> str | None:
        matches = list(self.box_pattern.finditer(solution))
        if not matches:
            return None

        match = matches[-1]
        start_idx = match.end()

        i = start_idx
        brace_depth = 1

        while i < len(solution):
            ch = solution[i]

            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1

            if brace_depth == 0:
                return solution[start_idx:i]

            i += 1

        return None

    def _get_reward(self, solution: str, answer: str) -> float | None:
        if type(answer) == int:
            answer = str(answer)
        
        logger.info(f"Original solution: {solution}")
        logger.info(f"Original answer: {answer}")

        solution = self._remove_think(solution)

        parsed_answer = self.extract_boxed_content(solution)
        if parsed_answer is None:
            return None

        parsed_answer = parsed_answer.strip()
        answer = answer.strip()

        return 1.0 if parsed_answer == answer else 0.0

    def __call__(self, input: VerifierInput) -> float | None:
        return self._get_reward(input.solution, input.answer)
