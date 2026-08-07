import abc

from .math_utils import extract_answer, grade_answer_verl


class Verifier(abc.ABC):
    @abc.abstractmethod
    def get_reward(self, solution: str, answer: str) -> float | None:
        raise NotImplementedError()

    def extract(self, solution: str) -> str | None:
        return None


class MathVerifier(Verifier):
    def __init__(self, strip_think: bool = True):
        self.strip_think = strip_think

    def _strip(self, solution: str) -> str:
        if self.strip_think:
            idx = solution.rfind("</think>")
            if idx != -1:
                solution = solution[idx + len("</think>") :]
        return solution

    def get_reward(self, solution: str, answer: str) -> float | None:
        score = grade_answer_verl(self._strip(solution), answer)
        return None if score is None else float(score)

    def extract(self, solution: str) -> str | None:
        return extract_answer(self._strip(solution))
