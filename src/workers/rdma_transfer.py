from functools import cached_property

import jax
import jax.experimental.multihost_utils as mh
import numpy as np
from jaxtyping import PyTree

EXPLICIT = jax.sharding.AxisType.Explicit


class RDMATransferServer:
    def __init__(self, train_workers: int):
        assert 1 <= train_workers <= self.max_rank, f"expected 1 <= train_workers <= {self.max_rank}"
        assert self.max_rank % train_workers == 0, (
            f"expected train workers to divide total workers, got {self.max_rank} and {train_workers}"
        )

        self.train_workers = train_workers
        self.n_slots = self.max_rank // train_workers
        self.slot_size = train_workers * jax.local_device_count()

        devices = np.array(jax.devices())

        self.shardings = [
            jax.NamedSharding(
                jax.make_mesh(
                    axis_shapes=(train_workers, jax.local_device_count()),
                    axis_names=("processes", "local_devices"),
                    axis_types=(EXPLICIT, EXPLICIT),
                    devices=devices[j * self.slot_size : (j + 1) * self.slot_size],  # type: ignore
                ),
                jax.P(),
            )
            for j in range(self.n_slots)
        ]

    def transfer(self, tree: PyTree) -> PyTree:
        shapes = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), tree)

        def make_buffer(s, sh):
            return jax.make_array_from_single_device_arrays(s.shape, sh, arrays=[], dtype=s.dtype)

        if self.slot == 0:
            tree = jax.device_put(tree, self.shardings[0])
        for i in range(self.n_slots - 1):
            src = self.shardings[i]
            buffer = tree if self.slot == i else jax.tree.map(lambda s: make_buffer(s, src), shapes)
            out = jax.device_put(buffer, self.shardings[i + 1])
            if self.slot == i + 1:
                tree = out

        tree = jax.tree.map(lambda x: x.block_until_ready(), tree)
        mh.sync_global_devices("rdma_transfer")
        return tree

    @cached_property
    def max_rank(self) -> int:
        return jax.process_count()

    @cached_property
    def rank(self) -> int:
        return jax.process_index()

    @cached_property
    def slot(self) -> int:
        return self.rank // self.train_workers

    @property
    def is_train_worker(self) -> bool:
        return self.rank < self.train_workers
