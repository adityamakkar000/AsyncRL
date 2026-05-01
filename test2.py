import jax
import jax.numpy as jnp
import numpy as np

# def _handle_array_process_allgather(x: jax.Array):
#     out =


if __name__ == "__main__":
    jax.distributed.initialize()

    ab = jnp.ones((2, 10))
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    ab = jax.device_put(ab, sharding)

    # print(isinstance(sharding, jax.NamedSharding))

    ab = jax.device_get(ab)

    train_devices = np.array(jax.devices())[4:]
    inference_devices = np.array(jax.devices())[:4]

    mesh = jax.make_mesh(
        (len(train_devices),), ("x",), axis_types=(jax.sharding.AxisType.Explicit,), devices=train_devices
    )
    mesh2 = jax.make_mesh(
        (len(inference_devices),), ("x",), axis_types=(jax.sharding.AxisType.Explicit,), devices=inference_devices
    )

    print(mesh)
    print(mesh2)

    sharding = jax.NamedSharding(mesh, jax.P())
    sharding2 = jax.NamedSharding(mesh2, jax.P())

    abc = jax.device_put(ab, sharding)
    abc2 = jax.device_put(ab, sharding2)

    print(abc.addressable_shards)
