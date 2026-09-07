from typing import Any, cast

import numpy as np
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.constants import SYSTEM_PROMPT

from .config import RLBatch


def compute_aux_metrics(batch: RLBatch) -> dict[str, float]:
    token_mask = batch.token_mask
    return {
        "mean_reward": np.mean(batch.rewards).item(),
        "std_reward": np.std(batch.rewards).item(),
        "max_reward": np.max(batch.rewards).item(),
        "min_reward": np.min(batch.rewards).item(),
        "mean_length": np.mean(token_mask.sum(axis=1)).item(),
        "median_length": np.median(token_mask.sum(axis=1)).item(),
    }


def apply_prompt_template(text: str) -> str:
    return f"""Solve the following math problem step by step. Put your answer inside \\boxed{{}}.

{text}

Remember to put your answer inside \\boxed{{}}."""


def resolve_pad_eos(tokenizer) -> tuple[int, int]:
    pad, eos = tokenizer.pad_token_id, tokenizer.eos_token_id
    assert eos is not None, f"tokenizer has no eos_token_id: {tokenizer}"
    return (eos if pad is None else pad), eos


def get_chat_template(system_prompt: bool, text: str) -> list[dict[str, str]]:
    chat = []
    if system_prompt:
        chat.append({"role": "system", "content": SYSTEM_PROMPT})
    chat.append({"role": "user", "content": apply_prompt_template(text)})
    return chat


def load_tokenizer(hf_model: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    assert isinstance(tokenizer, PreTrainedTokenizerBase), f"unexpected tokenizer type {type(tokenizer)}"
    return tokenizer


def decode_tokens(tokenizer: PreTrainedTokenizerBase, tokens: Any, skip_special_tokens: bool = False) -> str:
    return tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)


def apply_chat_template(
    tokenizer: PreTrainedTokenizerBase, system_prompt: bool, text: str, enable_thinking: bool = True
) -> list[int]:
    tokens = tokenizer.apply_chat_template(
        get_chat_template(system_prompt, text),
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        tokenize=True,
        return_dict=False,
    )
    return cast(list[int], tokens)


def pass_at_k(n: int, c: int, k: int) -> float:
    if k > n:
        return float(1.0 - (1.0 - c / n) ** k)
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))
