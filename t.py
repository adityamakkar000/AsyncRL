from src.data.utils import convert_rejection_samples_to_dataset

samples = convert_rejection_samples_to_dataset(
    "gs://arl-experiments/data/rejection_samples/aime_2025_10_samples.jsonl", 0.0, 1.0
)
breakpoint()
