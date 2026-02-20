import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import dataclass
from config import DatasetConfig
from dataloader import DataLoader
import re



@dataclass
class VerifierInput:
    solution: str
    answer: str

class Verifier:

    def __init__(self):
        self.box_start = re.compile(r"\\boxed\{")
        
    def extract_boxed_content(self, solution: str) -> str:
        match = self.box_start.search(solution)
        if not match:
            return None
        start_idx = match.end()
        i = start_idx
        cur_open = 1

        while i < len(solution):
            ch = solution[i]

            if ch == '{':
                cur_open += 1
            elif ch == '}':
                cur_open -= 1
            if cur_open == 0:
                return solution[start_idx:i]
            i += 1
        return None


    def _get_reward(self, solution: str, answer: str) -> float:
        parsed_answer = self.extract_boxed_content(solution)
        return 1.0 if (parsed_answer == answer) else 0.0

    def __call__(self, input: VerifierInput) -> float:
        return self._get_reward(input.solution, input.answer)

config = DatasetConfig(name="omnimath", batch_size=16, gcs_path="gs://arl-experiments/data/omnimath")

# test if this works for some examples: 
if __name__ == "__main__":
    verifier = Verifier()
    data_loader = DataLoader(config, seq_length=1024, hf_model="Qwen/Qwen3-4B")
    samples = data_loader(16)
    for sample in samples:
        input = VerifierInput(extracted_answer=sample.solution, ground_truth=sample.answer)
        print(verifier(input))

