import time
from functools import wraps
from typing import Any, Callable

import jax
from loguru import logger


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
        logger.info("Setting up {}...", component)
        start = time.time()
        setup_fn(*args, **kwargs)
        end = time.time()
        logger.info("{} setup complete in {:.2f} seconds.", component, end - start)
    return wrapper


class Key:
    def __init__(self, seed: int):
        self.key = jax.random.PRNGKey(seed)

    def __call__(self, num_keys: int = 1, split_by_process: bool = False):
        self.key, subkey = jax.random.split(self.key)
        if split_by_process:
            subkey = jax.random.fold_in(subkey, jax.process_index())
        return jax.random.split(subkey, (num_keys, 2))

