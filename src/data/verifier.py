import re
from dataclasses import dataclass


@dataclass
class VerifierInput:
    solution: str
    answer: str


class Verifier:
    def __init__(self):
        self.box_start = re.compile(r"\\boxed\{")

    # TODO: when we anneal, we will need to start extraction after </think>
    def extract_boxed_content(self, solution: str) -> str | None:
        match = self.box_start.search(solution)
        if not match:
            return None
        start_idx = match.end()
        i = start_idx
        cur_open = 1

        while i < len(solution):
            ch = solution[i]

            if ch == "{":
                cur_open += 1
            elif ch == "}":
                cur_open -= 1
            if cur_open == 0:
                return solution[start_idx:i]
            i += 1
        return None

    def _get_reward(self, solution: str, answer: str) -> float | None:
        parsed_answer = self.extract_boxed_content(solution)
        if parsed_answer is None or len(parsed_answer) == 0:
            return None
        return 1.0 if (answer in parsed_answer) else 0.0

    def __call__(self, input: VerifierInput) -> float | None:
        return self._get_reward(input.solution, input.answer)
