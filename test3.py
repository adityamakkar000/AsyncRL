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

        local_device_count = jax.local_device_count()
        n_devices = jax.device_count()

        self.shardings = dict()
        self.idexs = dict()

        for i, key in enumerate(["prev_idx", "current_idx", "next_idx"]):
            idx = (
                (self.stage - 1 + i) * train_workers * local_device_count,
                ((self.stage + i) * train_workers * local_device_count),
            )
            if idx[0] < 0 or idx[1] > n_devices:
                continue
            mesh = jax.make_mesh(
                axis_shapes=(train_workers, local_device_count),
                axis_names=("processes", "local_devices"),
                axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
                devices=devices[idx[0] : idx[1]],  # type: ignore
            )

            self.idexs[key] = idx
            self.shardings[key] = jax.NamedSharding(mesh, jax.P())

        print(f"rank : {self.rank}")
        print(f"step: {self.stage}")
        print(self.shardings)
        print(self.idexs)

    def transfer(self, x: PyTree):
        if isinstance(x, jnp.ndarray):
            return self._transfer_array(x)
        return jax.tree.map(self._transfer_array, x)

    def _transfer_array(self, x: Array):
        for i in range(self.max_rank // self.train_workers - 1):
            print(f"transfer step {i} for rank {self.rank}")
            self.step(i, x)
            mh.sync_global_devices(f"rdma_transfer_{i}")
        return x

    def step(self, i: int, x: Array):
        if i not in [self.stage, self.stage - 1]:
            return x

        x = x.copy()

        is_sender = i == self.stage

        buffer = (
            jax.device_put(x, self.shardings["current_idx"])
            if is_sender
            else jax.make_array_from_single_device_arrays(x.shape, self.shardings["prev_idx"], arrays=[], dtype=x.dtype)
        )

        buffer = buffer.block_until_ready()

        out = jax.device_put(buffer, self.shardings["next_idx"] if is_sender else self.shardings["current_idx"])

        out = out.block_until_ready()

        return out

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

train_workers = args.train_workers
print(f"using {train_workers} train workers")
assert 1 <= train_workers < jax.process_count(), "expected less than total"


train_mesh = jax.make_mesh(
    (train_workers * jax.local_device_count(),),
    ("devices",),
    axis_types=(jax.sharding.AxisType.Explicit,),
    devices=np.array(jax.devices())[: train_workers * jax.local_device_count()],
)

array = jnp.ones(
    (4 * GB // 1024,),
    dtype=jnp.float32,
)

# if jax.process_index() < train_workers:
#     array = jax.device_put(
#         array, jax.NamedSharding(train_mesh, jax.P('devices'))
#     )

transfer_server = RDMATransfer2(train_workers=train_workers)

out = transfer_server.transfer(array)


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
