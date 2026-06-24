from typing import List

import datasets

from src.data import grade_answer_verl
from src.data.math_utils import extract_boxed_answer


def boxed_score(doc: dict, results: List[str], extract_answer):
    answer: str = extract_answer(doc)
    scores = []
    for result in results:
        verl_score = grade_answer_verl(result, answer)
        scores.append(verl_score if verl_score else 0.0)
    average_score = sum(scores) / len(scores) if scores else 0.0

    return {"boxed_parse": average_score}


def process_gsm8k_results(doc: dict, results: List[str]) -> dict:
    def extract_answer(doc: dict) -> str:
        answer = doc.get("answer", "")
        return answer.split("####")[-1].strip()

    return boxed_score(doc, results, extract_answer)


def process_aime24_results(doc: dict, results: List[str]) -> dict:
    def extract_answer(doc: dict) -> str:
        return str(doc.get("Answer", ""))

    return boxed_score(doc, results, extract_answer)


def process_aime25_results(doc: dict, results: List[str]) -> dict:
    def extract_answer(doc: dict) -> str:
        return str(doc.get("answer", ""))

    return boxed_score(doc, results, extract_answer)


def process_math500_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": extract_boxed_answer(doc["solution"]),
        }
        return out_doc

    return dataset.map(_process_doc)


def process_math500_results(doc: dict, results: List[str]) -> dict:
    def extract_answer(doc: dict) -> str:
        return doc.get("answer", "")

    return boxed_score(doc, results, extract_answer)
