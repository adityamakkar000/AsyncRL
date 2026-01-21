
import jax.numpy as jnp
from jaxtyping import PyTree
from stax import modelBase


def sft_step(model : modelBase, params : PyTree,  batch : PyTree, train : bool):
    #TODO: implement
    return jnp.zeros((1,))

def standard_rl_step():
    pass

