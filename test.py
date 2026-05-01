import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import numpy as np

devices = np.array(jax.devices()).reshape(4, 2)
mesh = jax.sharding.Mesh(
    devices, ("data", "model"), axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit)
)
print(mesh.axis_names)


def test(a): ...


def test2(a):
    return jax.lax.while_loop(lambda x: x < 10000, lambda x: x + 1)
