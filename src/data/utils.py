import json
from typing import Any, cast

import gcsfs
import numpy as np
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.constants import SYSTEM_PROMPT

from .config import RLBatch, Sample


def load_jsonl_from_gcs(gs_prefix: str) -> list[dict]:
    """Load all JSONL files under gs_prefix (e.g. gs://bucket/data/omnimath/), return list of rows."""
    fs = gcsfs.GCSFileSystem()
    prefix = gs_prefix.replace("gs://", "") if gs_prefix.startswith("gs://") else gs_prefix
    if not prefix.endswith("/"):
        prefix += "/"

    rows: list[dict] = []
    if fs.exists(prefix):
        for path in fs.ls(prefix):
            if not path.endswith(".jsonl"):
                continue
            with fs.open(path, "r") as f:
                for line in f:
                    rows.append(json.loads(line.strip()))

    return rows


def upload_local_file_to_gcs(local_path: str, gs_path: str):
    """Upload a local file to GCS. gs_path should be like 'gs://bucket/path/file.jsonl'."""
    fs = gcsfs.GCSFileSystem()
    gcs_path = gs_path.replace("gs://", "") if gs_path.startswith("gs://") else gs_path

    with open(local_path, "rb") as src:
        data = src.read()
        with fs.open(gcs_path, "wb") as dst:
            dst.write(data)


def compute_aux_metrics(batch: RLBatch) -> dict[str, float]:
    token_mask = batch.reference_model_logprobs != -np.inf
    return {
        "mean_reward": np.mean(batch.rewards).item(),
        "std_reward": np.std(batch.rewards).item(),
        "max_reward": np.max(batch.rewards).item(),
        "min_reward": np.min(batch.rewards).item(),
        "mean_length": np.mean(token_mask.sum(axis=1)).item(),
        "median_length": np.median(token_mask.sum(axis=1)).item(),
    }


def filter_rejection_sampled_data(
    gcs_path: str, lower_bound: float | None = None, upper_bound: float | None = None
) -> list[dict]:
    assert lower_bound is not None or upper_bound is not None, (
        "At least one of lower_bound or upper_bound must be provided"
    )

    rows = load_jsonl_from_gcs(gcs_path)
    if lower_bound is not None:
        rows = [row for row in rows if row.get("pass_score", 0.0) >= lower_bound]
    if upper_bound is not None:
        rows = [row for row in rows if row.get("pass_score", 0.0) <= upper_bound]

    rows = sorted(rows, key=lambda x: x.get("pass_score", 0.0), reverse=True)
    return rows


def convert_rejection_samples_to_dataset(
    gcs_path: str, lower_bound: float | None = None, upper_bound: float | None = None
) -> list[Sample]:
    """Convert a list of RejectionSingleSample dicts to a list of Samples."""

    filtered_rejection_rows = filter_rejection_sampled_data(gcs_path, lower_bound, upper_bound)
    return [Sample.from_dict(row) for row in filtered_rejection_rows]


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
