import time
from functools import wraps
from typing import Any, Callable

import jax
from stax import staxLogger as logger


def setup(setup_fn: Callable[[Any], None], component: str):
    """
    Takes in a function that setups a component (e.g., dataset, model, optimizer)
    and returns a wrapped version that logs the time taken for setup.

    Args:
        setup_fn: The setup function to be wrapped. It should not return anything.
        component: A string representing the component being set up.
    Returns:
        A wrapped setup function that logs the time taken.
    """

    @wraps(setup_fn)
    def wrapper(*args, **kwargs):
        logger.info("Setting up {}", component)
        start = time.time()
        setup_fn(*args, **kwargs)
        end = time.time()
        logger.info(f"{component} setup complete in {end - start:.2f} seconds.")

    return wrapper


class Key:
    def __init__(self, seed: int):
        self.key = jax.random.PRNGKey(seed)

    def __call__(self, num_keys: int = 1, split_by_process: bool = False):
        self.key, subkey = jax.random.split(self.key)
        if split_by_process:
            subkey = jax.random.fold_in(subkey, jax.process_index())
        return jax.random.split(subkey, (num_keys, 2))


def set_jax_cache(path: str):
    """Sets the JAX cache directory to the specified path."""
    jax.config.update("jax_compilation_cache_dir", path)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

