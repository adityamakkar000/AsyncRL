from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.multihost_utils import broadcast_one_to_all, sync_global_devices

# from stax import Tracker


def pallas_call(): ...


def custom_broadcast(global_mesh):
    @jax.jit
    def transfer():
        @partial(
            jax.shard_map,
            mesh=global_mesh,
            in_shardings=jax.NamedSharding(global_mesh, jax.P()),
            out_shardings=jax.NamedSharding(global_mesh, jax.P("dp")),
        )
        def _transfer(): ...

        return transfer()


def train_sync_weights(params, train_mesh):
    print("Syncing weights across train workers...")
    sync_global_devices("weightSync")
    print("gathering")

    params_gathered = jax.jit(lambda x: x, out_shardings=jax.NamedSharding(train_mesh, jax.P()))(params)
    params_cpu = jax.device_get(params_gathered)
    _params = broadcast_one_to_all(params_cpu)


def inference_sync_weights(params_cpu, inference_mesh):
    print("Syncing weights across inference workers...")
    sync_global_devices("weightSync")

    params_cpu = broadcast_one_to_all(params_cpu)

    print("Broadcasted params to all inf workers: {}".format(params_cpu.sum()))
    return params_cpu


def main():
    train_workers = 1
    inference_workers = 1

    print(f"Process index: {jax.process_index()}, total processes: {jax.process_count()}")

    n = jax.device_count() // 2

    train_devices = np.array(jax.devices())[:n]
    inference_devices = np.array(jax.devices())[n:]

    train_mesh = jax.make_mesh((n,), ("dp",), axis_types=(jax.sharding.AxisType.Explicit,), devices=train_devices)
    inference_mesh = jax.make_mesh(
        (n,), ("dp",), axis_types=(jax.sharding.AxisType.Explicit,), devices=inference_devices
    )

    assert train_workers + inference_workers == jax.process_count(), (
        "Total number of workers must equal the number of JAX processes"
    )

    shape = (32768, 32768)

    if jax.process_index() < 1:
        pspec = jax.P("dp")
        params = jax.device_put(jnp.ones(shape), jax.NamedSharding(train_mesh, pspec))

        train_sync_weights(params, train_mesh)
    else:
        params = np.zeros(shape)
        inference_sync_weights(params, inference_mesh)

    sync_global_devices("done")


if __name__ == "__main__":
    jax.distributed.initialize()
    name = "test2"
    if jax.process_index() == 0:
        with jax.profiler.trace(f"gs://arl-experiments/profile/weightSync/{name}"):
            main()
    else:
        main()
