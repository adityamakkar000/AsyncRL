from src.constants import DATA, GS_BUCKET
from src.data import Sample, load_jsonl_from_gcs


def fetch_eval_samples(task: str) -> list[Sample]:
    """Fetches evaluation prompts for a given task."""
    gs_path = f"{GS_BUCKET}/{DATA}/{task}"
    rows = load_jsonl_from_gcs(gs_path)
    if not rows:
        raise ValueError(f"No rows found at {gs_path}")
    samples: list[Sample] = [Sample.from_dict(r) for r in rows]
    return samples
