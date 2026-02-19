"""
Register datasets below. Each dataset should be a function that returns a list of samples.
"""

from src.data.config import DataConfig, Sample
from src.data.utils import register_dataset, DATASET_DICT

@register_dataset("omnimath")
def process_omnimath():
  return DATASET_DICT["omnimath"]()



