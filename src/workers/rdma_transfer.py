import itertools
import time
from functools import cached_property

import jax
import jax.experimental.multihost_utils as mh
import numpy as np
from jaxtyping import PyTree
from stax.logger import staxLogger as logger

EXPLICIT = jax.sharding.AxisType.Explicit


def allocate_buffer(shapes: PyTree, sharding: jax.NamedSharding):
    return jax.tree.map(
        lambda s: jax.make_array_from_single_device_arrays(s.shape, sharding=sharding, arrays=[], dtype=s.dtype), shapes
    )


def get_default_sharding(mesh: jax.sharding.Mesh) -> jax.NamedSharding:
    return jax.NamedSharding(mesh, jax.P())


def create_mesh(devices: np.ndarray) -> jax.sharding.Mesh:
    assert devices.ndim == 1, "devices must be a 1-dimensional array"
    return jax.make_mesh(
        axis_shapes=(devices.size,), axis_names=("devices",), axis_types=(EXPLICIT,), devices=tuple(devices)
    )


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
        self.meshes = [create_mesh(devices[j * self.slot_size : (j + 1) * self.slot_size]) for j in range(self.n_slots)]

    def get_zero_sharding(self, mesh: jax.sharding.Mesh) -> jax.NamedSharding:
        devices = np.array(mesh.devices).reshape(-1)
        mesh = create_mesh(devices)
        return get_default_sharding(mesh)

    def transfer(self, tree: PyTree, init_mesh: jax.sharding.Mesh | None = None) -> PyTree:
        mh.sync_global_devices("rdma_transfer_start")
        shapes = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), tree)
        shardings = list(map(get_default_sharding, self.meshes))

        if init_mesh is not None:
            if not self.slot == 0:
                raise ValueError("only train workers should pass and init mesh")
            shardings[0] = self.get_zero_sharding(init_mesh)

        t0 = time.perf_counter()
        logger.info(f"[rdma] slot {self.slot} transfer start", log_for_all=True)
        if self.slot == 0:
            tree = jax.device_put(tree, shardings[0])
            logger.info(
                f"[rdma] slot {self.slot} local device_put done {time.perf_counter() - t0:.1f}s", log_for_all=True
            )

        for i, (src, dest) in enumerate(itertools.pairwise(shardings)):
            buffer = tree if self.slot == i else allocate_buffer(shapes, src)
            logger.info(
                f"[rdma] slot {self.slot} hop {i} buffer ready {time.perf_counter() - t0:.1f}s", log_for_all=True
            )
            out = jax.device_put(buffer, dest)
            logger.info(
                f"[rdma] slot {self.slot} hop {i} device_put issued {time.perf_counter() - t0:.1f}s", log_for_all=True
            )
            if self.slot == i + 1:
                tree = out

        tree = jax.tree.map(lambda x: x.block_until_ready(), tree)
        logger.info(f"[rdma] slot {self.slot} block_until_ready done {time.perf_counter() - t0:.1f}s", log_for_all=True)
        mh.sync_global_devices("rdma_transfer_end")
        logger.info(
            f"[rdma] slot {self.slot} sync_global_devices done {time.perf_counter() - t0:.1f}s", log_for_all=True
        )
        return tree

    def get_mesh(self, index: int = 0) -> jax.sharding.Mesh:
        if index < 0 or index > len(self.meshes):
            raise ValueError(f"expected index in 0 to {len(self.meshes)}, got {index}")
        return self.meshes[index]

    def get_sharding(self, index: int = 0) -> jax.sharding.NamedSharding:
        return get_default_sharding(self.get_mesh(index))

    @cached_property
    def max_rank(self) -> int:
        return jax.process_count()

    @cached_property
    def rank(self) -> int:
        return jax.process_index()

    @cached_property
    def slot(self) -> int:
        return self.rank // self.train_workers
