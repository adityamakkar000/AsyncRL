from typing import List

from src.data import grade_answer_verl


def gsm8k_extract_answer(doc: dict) -> str:
    answer = doc.get("answer", "")
    return answer.split("####")[-1].strip()


def process_gsm8k_results(doc: dict, results: List[str]) -> dict:
    answer = gsm8k_extract_answer(doc)
    scores = []
    for result in results:
        verl_score = grade_answer_verl(result, answer)
        scores.append(verl_score if verl_score else 0.0)
    average_score = sum(scores) / len(scores) if scores else 0.0

    return {"boxed_parse": average_score}
