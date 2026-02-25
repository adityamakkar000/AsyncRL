import re
from dataclasses import dataclass


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
        solution = self._remove_think(solution)

        parsed_answer = self.extract_boxed_content(solution)
        if parsed_answer is None:
            return None

        parsed_answer = parsed_answer.strip()
        answer = answer.strip()

        return 1.0 if parsed_answer == answer else 0.0

    def __call__(self, input: VerifierInput) -> float | None:
        return self._get_reward(input.solution, input.answer)


test_str = "<|im_start|>user\nSolve the following math problem step by step. Put your answer inside \\boxed{}.\nWhich of the following integers cannot be written as a product of two integers, each greater than 1: 6, 27, 53, 39, 77?\nRemember to put your answer inside \\boxed{}.<|im_end|>\n<|im_start|>assistant<think>\nOkay, so I need to figure out which of these numbers—6, 27, 53, 39, 77—cannot be written as a product of two integers, each greater than 1. Let me start by recalling what it means for a number to be expressible as a product of two integers greater than 1. \n\nFirst, let me list the numbers again: 6, 27, 53, 39, 77. I need to check each one to see if they can be broken down into two factors, both of which are integers greater than 1. If a number can be written as such, then it's not the answer; if not, then it is the answer.\n\nStarting with 6. Let me think. 6 can be written as 2 × 3, both of which are greater than 1. So 6 is okay. \n\nNext is 27. Let me try factors. The factors of 27 are 1, 3, 9, 27. So, if I take 3 × 9, both are greater than 1. So 27 can be written as a product of two integers greater than 1. So 27 is okay.\n\nThen 53. Hmm, 53 is a prime number, right? Wait, is 53 prime? Let me check. The square root of 53 is about 7.28. So I need to check divisibility by primes up to 7. 2? No, it's odd. 3? 3×17 is 51, 3×18 is 54. 5? Doesn't end with 0 or 5. 7? 7×7 is 49, 7×8 is 56. So no, 53 isn't divisible by any of these. Therefore, 53 is a prime number. So its only factors are 1 and itself. Therefore, it cannot be written as a product of two</think>\n\nTo determine which of the given integers cannot be written as a product of two integers, each greater than 1, we analyze each number in the list:\n\n- **6**: Can be factored as $2 \\times 3$, both greater than 1. ✅\n- **27**: Can be factored as $3 \\times 9$, both greater than 1. ✅\n- **53**: 53 is a prime number. Its only factors are 1 and 53. Since neither of these is greater than 1, **53 cannot** be written as a product of two integers greater than 1. ❌\n- **39**: Can be factored as $3 \\times 13$, both greater than 1. ✅\n- **77**: Can be factored as $7 \\times 11$, both greater than 1. ✅\n\n---\n\nTherefore, the integer that **cannot** be written as a product of two integers greater than 1 is:\n\n$$\n\\boxed{53}\n$$<|im_end|>"
answer = "53"


verifier = Verifier()
reward = verifier(VerifierInput(solution=test_str, answer=answer))

breakpoint()
