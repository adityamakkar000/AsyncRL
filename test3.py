import time

import jax
import numpy as np
from jax.experimental.multihost_utils import sync_global_devices


def train_worker(mesh, num_steps):
    print("waiting for barrier")
    sync_global_devices("trainDone")
    print("after barrier")


def inference_worker(mesh):
    for i in range(100):
        print(i)
        time.sleep(0.5)
    sync_global_devices("trainDone")


def main():
    jax.distributed.initialize()
    print("started with a total of {} devices".format(jax.device_count()))
    rank = jax.process_index()

    n = jax.device_count() // 2

    train_devices = np.array(jax.devices())[:n]
    inference_devices = np.array(jax.devices())[n:]

    train_mesh = jax.make_mesh((n,), ("data",), axis_types=(jax.sharding.AxisType.Explicit,), devices=train_devices)
    inference_mesh = jax.make_mesh(
        (n,), ("data",), axis_types=(jax.sharding.AxisType.Explicit,), devices=inference_devices
    )

    if rank == 0:
        train_worker(train_mesh, 100000)
    else:
        inference_worker(inference_mesh)

    print("done")
    # sync_global_devices("trainDone")


if __name__ == "__main__":
    import sys

    sys.exit(0)
    main()
