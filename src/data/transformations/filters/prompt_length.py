from typing import Any

from transformers import AutoTokenizer

from src.model import BaseModel

from ...envs import Environment


class PromptLengthFilter:
    def __init__(self, max_length: int, hf_model: str) -> None:
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)

    def __call__(self, problem: Any, env: Environment) -> bool:
        tokens = BaseModel.apply_chat_template(
            self.tokenizer, env.system_prompt, env.user_prompt(env.question(problem))
        )
        return len(tokens) <= self.max_length
