import contextlib
import socket
import time
from functools import partial

import jax
import jax.experimental.multihost_utils as mh
import jax.numpy as jnp
import numpy as np
from jax.experimental.transfer import start_transfer_server

jax.distributed.initialize()

GB = 1024**3
block_size = 100 * 1024 * 1024  # 1 MB
total_bytes = 16 * GB  # 1 GB


@contextlib.contextmanager
def timer(name: str):
    mh.sync_global_devices(f"start_{name}")
    print(f"{name} starting...")
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    mh.sync_global_devices(f"end_{name}")
    print(f"{name} elapsed time: {end - start:.6f} seconds")
    print(f"{name} throughput: {total_bytes / (end - start) / GB:.6f} GB/s")


rank = jax.process_index()

# 1 GB array
# make a dict of arrays of blockk_size up to total_bytes
blocks = {}
for i in range(0, int(total_bytes // block_size)):
    n_bytes = i * block_size
    blocks[i] = rank * np.ones(
        (block_size // 4,),  # float32 has 4 bytes
        dtype=np.float32,
    )

print("Blocks initialized with shape:", {k: v.shape for k, v in blocks.items()})
print("Blocks initialized with dtype:", {k: v.dtype for k, v in blocks.items()})

mesh = jax.make_mesh(
    (jax.process_count(), jax.local_device_count()),
    ("processes", "local_devices"),
    axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
)

local_mesh = jax.make_mesh(
    (jax.local_device_count(),),
    ("local_devices",),
    axis_types=(jax.sharding.AxisType.Explicit,),
    devices=np.array(jax.local_devices()),
)


# (4, 64)
# (4, 64)
# (4, 64)
# (4, 64)

# (16, 64)


@partial(jax.jit, static_argnames=("mesh",))
def rdma_transfer(x, mesh):
    @partial(
        jax.shard_map,
        mesh=mesh,
        out_specs=jax.P(),
        in_specs=jax.P("processes"),
    )
    def _transfer(x):
        idx = jax.lax.axis_index("processes")
        x = jax.lax.cond(
            idx == 0,
            lambda x: x,  # Source process returns the input array
            lambda x: jnp.zeros_like(x),  # Other processes return zeros
            x,
        )
        x = jax.lax.psum(x, axis_name="processes")
        return x

    return jax.tree.map(_transfer, x)


with timer("Host local array to global array"):
    a_shard = jax.tree.map(lambda x: mh.host_local_array_to_global_array(x, mesh, jax.P("processes")), blocks)
    a_shard = jax.tree.map(lambda x: x.block_until_ready(), a_shard)

# rdma test 1
# print("warming up")
# warmup_steps = 2
# start = time.perf_counter()
# for _ in range(warmup_steps):
#     a = mh.broadcast_one_to_all(a, is_source=(rank == 0))
# end = time.perf_counter()
# print("Warmup elapsed time:", end - start)

# print("measuring broadcast time")
# start = time.perf_counter()
# a = mh.broadcast_one_to_all(a, is_source=(rank == 0))
# end = time.perf_counter()
# print("Elapsed time:", end - start)
# print("GB/s:", (a.nbytes / (end - start)) / (1024 ** 3))
# print(a)


# weight transfer test

warmup_steps = 2
with timer("RDMA transfer warmup"):
    for _ in range(warmup_steps):
        a_out = rdma_transfer(a_shard, mesh)
        a_out = jax.tree.map(lambda x: x.block_until_ready(), a_out)

with timer("RDMA transfer"):
    a_out = rdma_transfer(a_shard, mesh)
    a_out = jax.tree.map(lambda x: x.block_until_ready(), a_out)


def get_current_vm_internal_ip():
    return socket.gethostbyname(socket.gethostname())


def setup_transfer_server(local_ip: str, port: int):
    backend_client = jax.devices()[0].client
    server = start_transfer_server(
        backend_client,
        f"{local_ip}:{port}",
        [f"{local_ip}:0"] * jax.device_count(),
    )
    return server


main_ip = "10.128.0.3"
ip = get_current_vm_internal_ip()
server = setup_transfer_server(ip, 8000)

if ip != main_ip:
    client = server.connect(main_ip + ":8000")

with timer("host to local array"):
    local_a = jax.tree.map(lambda x: jax.device_put(x, jax.NamedSharding(local_mesh, jax.P())), blocks)
    local_a = jax.tree.map(lambda x: x.block_until_ready(), local_a)
shape_type = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), local_a)
with timer("DCN transfer"):
    if ip == main_ip:
        server.await_pull(0, local_a)
    else:
        new_params = jax.tree.map(lambda x: x.block_until_ready(), client.pull(0, shape_type))
    mh.sync_global_devices("transfer_done")
