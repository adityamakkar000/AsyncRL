import re
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

ESCAPED_LATEX = [
    (re.compile(r"\x0c(?=rac|loor|orall|box|rown)"), r"\\f"),
    (re.compile(r"\x0c"), ""),
    (re.compile(r"\x0b(?=ec|arphi|artheta|artriangle|arepsilon|ert|dots|space)"), r"\\v"),
    (re.compile(r"\x0b"), ""),
    (re.compile(r"\t(?=heta|herefore|an\b|anh|au\b|ext|frac|imes|o\b|op\b|riangle)"), r"\\t"),
]


def fix_latex(text: str) -> str:
    for pattern, repl in ESCAPED_LATEX:
        text = pattern.sub(repl, text)
    return text


class FixEscapedLatex:
    def __init__(self, fields: Sequence[str] = ("question", "solution")) -> None:
        self.fields = fields

    def __call__(self, problems: list, question: Callable[[Any], str]) -> list:
        return [self.fix(problem) for problem in problems]

    def fix(self, problem: Any) -> Any:
        values = {field: getattr(problem, field) for field in self.fields}
        return replace(problem, **{k: fix_latex(v) for k, v in values.items() if v is not None})
