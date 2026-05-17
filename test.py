import socket
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.clusters.cloud_tpu_cluster import GceTpuCluster
from jax.experimental.transfer import start_transfer_server


def get_current_vm_internal_ip():
    return socket.gethostbyname(socket.gethostname())


@lru_cache
def get_global_ip():
    return GceTpuCluster.get_coordinator_address(60).split(":")[0]


if __name__ == "__main__":
    jax.distributed.initialize()

    ip = get_current_vm_internal_ip()
    backend_client = jax.devices()[0].client
    server = start_transfer_server(
        backend_client,
        f"{ip}:8000",  # Random port binding
        [f"{ip}:0"] * jax.device_count(),
    )

    mesh = jax.make_mesh(
        (4,), ("x",), axis_types=(jax.sharding.AxisType.Explicit,), devices=np.array(jax.local_devices())
    )

    sharding = jax.NamedSharding(mesh, jax.P("x"))
    ab = jax.device_put(jnp.arange(4), sharding) + jax.process_index()
    ab_shape = jax.ShapeDtypeStruct((4,), dtype=jnp.int32, sharding=sharding)

    if ip != get_global_ip():
        client = server.connect(get_global_ip() + ":8000")

    breakpoint()
