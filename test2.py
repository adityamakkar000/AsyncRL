import argparse
from functools import lru_cache

import jax
import jax.experimental.multihost_utils as mh
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

parser = argparse.ArgumentParser()

parser.add_argument("--train-workers", dest="train_workers", type=int, required=False)


class RDMATransfer2:
    def __init__(self, train_workers: int):
        assert self.max_rank % train_workers == 0, (
            f"expected train workers to divide total workers get {self.max_rank} and {train_workers}"
        )

        self.train_workers = train_workers

        devices = np.array(jax.devices())

        send_idx = ((self.stage * train_workers) % self.max_rank, ((self.stage + 1) * train_workers) % self.max_rank)

        recieve_idx = (
            ((self.stage + 1) * train_workers) % self.max_rank,
            ((self.stage + 2) * train_workers) % self.max_rank,
        )

        send_devices = devices[send_idx[0] : send_idx[1]]
        receive_devices = devices[recieve_idx[0] : recieve_idx[1]]

        self.send_mesh = jax.make_mesh(
            axis_shapes=(train_workers, jax.local_device_count()),
            axis_names=("processes", "local_devices"),
            axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
            devices=send_devices,  # type: ignore
        )
        self.recive_mesh = jax.make_mesh(
            axis_shapes=(train_workers, jax.local_device_count()),
            axis_names=("processes", "local_devices"),
            axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
            devices=receive_devices,  # type: ignore
        )

        self.send_sharding = jax.NamedSharding(self.send_mesh, jax.P())

        self.recive_sharding = jax.NamedSharding(self.recive_mesh, jax.P())

    def transfer(self, x: PyTree):
        if isinstance(x, jnp.ndarray):
            return self._transfer_array(x)
        return jax.tree.map(self._transfer_array, x)

    def _transfer_array(self, x: Array):
        for i in range(self.max_rank // self.train_workers - 1):
            x = self.step(i, x)
            mh.sync_global_devices(f"rdma_transfer_{i}")
        return x

    def step(self, i: int, x: Array):
        if i not in [self.stage, self.stage + 1]:
            return x

        if i == self.stage:
            buffer = jax.device_put(x, self.send_sharding)
            return x

        if i == self.stage + 1:
            buffer = jax.make_array_from_single_device_arrays(x.shape, self.send_sharding, arrays=[], dtype=x.dtype)

        buffer = jax.device_put(buffer, self.recive_sharding)

        return x if self.stage == i else buffer

    @property
    @lru_cache
    def max_rank(self):
        return jax.process_count()

    @property
    @lru_cache
    def rank(self):
        return jax.process_index()

    @property
    @lru_cache
    def group(self):
        return

    @property
    def is_train_worker(self):
        return self.rank < self.train_workers

    @property
    @lru_cache
    def stage(self):
        return self.rank // self.train_workers


args = parser.parse_args()

jax.distributed.initialize()

GB = 1024**3
block_size = 100 * 1024 * 1024  # 1 MB
total_bytes = 16 * GB  # 1 GB

rank = jax.process_index()

# 1 GB array
# make a dict of arrays of blockk_size up to total_bytes
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


devices = np.array(jax.devices())
train_workers = args.train_workers
print(f"using {train_workers} train workers")
assert 1 <= train_workers < jax.process_count(), "expected less than total"

train_devices = devices[: train_workers * jax.local_device_count()]
inference_devices = devices[train_workers * jax.local_device_count() :]

train_mesh = jax.make_mesh(
    (train_workers, jax.local_device_count()),
    ("processes", "local_devices"),
    axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    devices=train_devices,
)

inference_mesh = jax.make_mesh(
    (jax.process_count() - train_workers, jax.local_device_count()),
    ("processes", "local_devices"),
    axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    devices=inference_devices,
)
array = jnp.ones(
    (4, GB),
    dtype=jnp.float32,
)

train_sharding = jax.NamedSharding(train_mesh, jax.P())
inference_sharding = jax.NamedSharding(mesh, jax.P())

if rank < train_workers:
    buffer = jax.device_put(array, train_sharding)
else:
    buffer = jax.make_array_from_single_device_arrays(
        array.shape, sharding=train_sharding, arrays=[], dtype=jnp.float32
    )

# array_sharded = jax.device_put(
#     buffer,
#     inference_sharding
# )

array_sharded = jax.device_put(buffer, inference_sharding)

breakpoint()

# train ... [0,4] [4,8]
# inference ... [4,8], [8,12]
# inference ... [8,12], [12,16]
# inference ... [12, 16], [0, 4]


# train ... [0, 8] [8, 16]
# train ... [0, 8] [8, 16]
# inference ... [8, 16], [0, 8]
#


# class RDMATransfer:

#     @dataclass
#     class ShardingPair:
#         mesh: jax.sharding.Mesh
#         sharding: jax.NamedSharding

#         @classmethod
#         def make_pair(self, send_devices, receive_devices):
#             mesh = jax.make_mesh(
#                 axis_shapes=(rank, jax.local_device_count()),
#                 axis_names=("processes", "local_devices"),
#                 axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
#                 devices=devices # type: ignore
#             )

#             sharding = jax.NamedSharding(
#                 mesh,
#                 jax.P()
#             )

#             return RDMATransfer.ShardingPair(mesh=mesh, sharding=sharding)

#     def __init__(self, train_workers: int):
#         self.train_workers = train_workers
#         self.devices = np.array(jax.devices())
#         self.tree_shardings : dict[int, RDMATransfer.ShardingPair] = dict()

#         self.build_meshes()

#     def build_meshes(self):
#         i = 2
#         local_devices = jax.local_device_count()
#         while i <= self.max_rank:
#             send_devices = self.devices[:i*local_devices]
#             send_mesh= jax.make_mesh(
#                 axis_shapes=(i, local_devices),
#                 axis_names=("processes", "local_devices"),
#                 axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
#                 devices=current_devices # type: ignore
#             )

#             sharding = jax.NamedSharding(
#                 current_mesh,
#                 jax.P()
#             )

#             self.tree_shardings[i] = RDMATransfer.ShardingPair.make_pair(i, send_devices)

#             i *= 2

#     def make_sharding(self, i:int, devices: np.ndarray):
#         return jax.make_mesh(
#             axis_shapes=(i, jax.local_device_count()),
#             axis_names=("processes", "local_devices"),
#             axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
#             devices=devices # type: ignore
#         )


#     def transfer(self, x: PyTree):
#         if isinstance(x, jnp.ndarray):
#             return self._transfer_array(x)
#         return jax.tree.map(self._transfer_array, x)

#     def _transfer_array(self, x: Array):
#         for i, sharding_pair in self.tree_shardings.items():
#             if self.rank < i:
#                 x = jax.device_put(x, sharding_pair.sharding)
#             else:
#                 x = jax.make_array_from_single_device_arrays(
#                     x.shape,
#                     sharding=sharding_pair.sharding,
#                     arrays=[],
#                     dtype=x.dtype
#                 )

#     @property
#     @lru_cache
#     def max_rank(self):
#         return jax.process_count()

#     @property
#     @lru_cache
#     def rank(self):
#         return jax.process_index()

#     @property
#     def is_train_worker(self):
#         return self.rank < self.train_workers

# option a) do the thing and try pallas
# not sure if this works
# option b) try to make 1-1 meshes and transfer that way?
# i don't think b) works because the mesh must be contingous i belive?

# process 0
# process 1
# process 2
# process 3
