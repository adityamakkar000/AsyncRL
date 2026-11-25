import os
import re
from functools import cache

import httpx

URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwen3.7-flash"
TIMEOUT = 2.0
API_KEY = "OPENROUTER_API_KEY"
VERDICT = re.compile(r"\b(INCORRECT|CORRECT)\b")

PROMPT = """You are checking a student's final answer against a reference answer.

<student_answer>{given}</student_answer>
<reference_answer>{expected}</reference_answer>

Rules:
- Reply CORRECT only if the student answer is a specific value that is mathematically equal to the reference answer. Different formatting, units, or a multiple-choice letter written next to the correct value are fine.
- Reply INCORRECT if the student answer is empty, a placeholder or variable name, text that is not a value, more than one candidate answer, or a different value.

Reply with exactly one word: CORRECT or INCORRECT."""


def require_api_key() -> None:
    if not os.environ.get(API_KEY):
        raise RuntimeError(f"llm_judge is enabled but {API_KEY} is not set")


@cache
def client() -> httpx.Client:
    return httpx.Client(headers={"Authorization": f"Bearer {os.environ[API_KEY]}"}, timeout=TIMEOUT)


def ask(prompt: str) -> bool:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning": {"enabled": False},
        "temperature": 0,
        "max_tokens": 8,
    }
    try:
        response = client().post(URL, json=body)
        response.raise_for_status()
        text = response.json()["choices"][0]["message"].get("content") or ""
    except (httpx.HTTPError, KeyError, IndexError, ValueError):
        return False
    verdicts = VERDICT.findall(text.upper())
    return bool(verdicts) and verdicts[-1] == "CORRECT"


def judge_answer(given: str, expected: str) -> bool:
    return ask(PROMPT.format(given=given, expected=expected))
