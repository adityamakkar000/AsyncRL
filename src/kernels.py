from functools import partial

import jax


def broadcast_to_inference_workers(params):
    @jax.jit
    def send():
        @partial(jax.shard_map, out_specs=jax.P())
        def _send(): ...

        return _send()
