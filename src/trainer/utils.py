import time
from functools import wraps
from typing import Any, Callable

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

