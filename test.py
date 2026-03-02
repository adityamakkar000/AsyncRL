import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp

devices = jax.devices()
mesh = jax.make_mesh((len(devices),), ("dp",))

jax.set_mesh(mesh)

new_mesh = jax.make_mesh((len(devices),), ("dp2",))
sharding = jax.NamedSharding(new_mesh, jax.sharding.PartitionSpec())
x = jnp.arange(10)

x_sharded = jax.device_put(x, sharding)

x_sharded_sliced = jax.lax.dynamic_index_in_dim(x_sharded, 0, keepdims=False)
with jax.set_mesh(new_mesh):
    x_sharded_sliced_2 = jax.lax.dynamic_index_in_dim(x_sharded, 0, keepdims=False)

print(x_sharded.sharding)
print(x_sharded_sliced.sharding)
print(x_sharded_sliced_2.sharding)

breakpoint()
