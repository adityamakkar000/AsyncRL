import hashlib
import json

from omegaconf import DictConfig, OmegaConf

from .config import evalConfig


def hash_dictConfig(d: DictConfig | evalConfig) -> str:
    """
    Hash a DictConfig object.
    Args:
        d (DictConfig): The DictConfig object to hash.
    Returns:
        str: The SHA-256 hash of the DictConfig.
    """

    # TODO:
    # find a way to only include some keys

    hash_obj = OmegaConf.to_container(
        d,
        resolve=True,
        throw_on_missing=True,
    )
    hash_dict = json.dumps(hash_obj, sort_keys=True)
    return hashlib.sha256(hash_dict.encode("utf-8")).hexdigest()
