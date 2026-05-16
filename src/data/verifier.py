from .math_utils import grade_answer_verl


def _remove_think(solution: str) -> str:
    end_tag = "</think>"
    idx = solution.rfind(end_tag)
    if idx == -1:
        return solution
    return solution[idx + len(end_tag) :]


def verl_score(solution: str, answer: str) -> float | None:
    verl_score = grade_answer_verl(solution, answer)
    if verl_score is None:
        return None
    return 1.0 if verl_score else 0.0


class Verifier:
    @staticmethod
    def get_reward(solution: str, answer: str) -> float | None:
        solution = _remove_think(solution)
        return verl_score(solution, answer)
