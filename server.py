import queue
import socket
import threading
import time
from dataclasses import dataclass
from multiprocessing.managers import BaseManager

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.clusters.cloud_tpu_cluster import GceTpuCluster
from jax.experimental.multihost_utils import broadcast_one_to_all, sync_global_devices


@dataclass
class AsyncConfig:
    train_workers: int
    inference_workers: int
    prompt_queue: queue.Queue
    rollout_queue: queue.Queue
    weight_sync_queue: queue.Queue
    train_mesh: jax.sharding.Mesh
    inference_mesh: jax.sharding.Mesh


@dataclass
class AsyncState:
    MRUparams: jax.Array
    updated: bool


class QueueManager(BaseManager):
    pass


PORT = 50000
KEY = b"test"


def get_current_vm_internal_ip():
    return socket.gethostbyname(socket.gethostname())


def get_global_ip():
    return GceTpuCluster.get_coordinator_address(60).split(":")[0]


def start_server():
    def start_server_backend():
        manager = QueueManager(address=("0.0.0.0", PORT), authkey=KEY)
        server = manager.get_server()
        print(f"[Server] Queue server listening on {get_global_ip()}:{PORT}...")
        server.serve_forever()

    server_thread = threading.Thread(target=start_server_backend, daemon=True)
    server_thread.start()


def get_queue(server_ip: str):
    manager = QueueManager(address=(server_ip, PORT), authkey=KEY)

    for _ in range(6):
        try:
            manager.connect()
            return manager.get_prompt_queue(), manager.get_rollout_queue(), manager.get_weight_sync_queue()
        except ConnectionError:
            print(f"[Client] Waiting for server at {server_ip}...")
            time.sleep(1)

    raise ConnectionError(f"Could not connect to server at {server_ip} after multiple attempts.")


def train_sync_weights(params, async_config: AsyncConfig):
    for _ in range(async_config.inference_workers):
        async_config.weight_sync_queue.put("sync")

    print(f"Placed sync signals for {async_config.inference_workers} inference workers.")
    while not async_config.weight_sync_queue.empty():
        time.sleep(0.1)
    sync_global_devices("weightSync")

    params_gathered = jax.jit(lambda x: x, out_shardings=jax.NamedSharding(async_config.train_mesh, jax.P()))(params)
    params_cpu = jax.device_get(params_gathered)
    print("Gathered params")
    sync_global_devices("gathered")

    _params = broadcast_one_to_all(params_cpu)


def train_fn(async_config: AsyncConfig):
    params = jnp.ones((12, 12))
    sharding = jax.NamedSharding(async_config.train_mesh, jax.P("dp"))
    params = jax.device_put(params, sharding)

    train_sync_weights(params, async_config)

    for i in range(10):
        print(async_config.prompt_queue.get())
        params += 1  # simulate training by updating the params
        train_sync_weights(params, async_config)


def inference_sync_weights(params_cpu, async_config: AsyncConfig):
    async_config.weight_sync_queue.get()
    while not async_config.weight_sync_queue.empty():
        time.sleep(0.1)
    sync_global_devices("weightSync")
    sync_global_devices("gathered")

    params_cpu = broadcast_one_to_all(params_cpu)
    print("Broadcasted params to all inf workers")
    return params_cpu


def monitor_weight_sync(state: AsyncState, async_config: AsyncConfig):
    while True:
        if async_config.weight_sync_queue.full():
            state.MRUparams = inference_sync_weights(state.MRUparams, async_config)
            state.updated = True
            print("Updated weights on inference worker")

        time.sleep(0.1)


def _maybe_update_params(params, state: AsyncState) -> jax.Array:
    new_params = params
    sharding = params.sharding
    if state.updated:
        state.updated = False
        new_params = state.MRUparams
        print("New params havse been updated to MRU params")
    new_params = jax.device_put(new_params, sharding)

    return new_params


def inference_fn(async_config: AsyncConfig):
    params = jnp.zeros((12, 12))
    local_mesh = jax.make_mesh(
        (jax.local_device_count(),), ("dp",), axis_types=(jax.sharding.AxisType.Explicit,), devices=jax.local_devices()
    )
    sharding = jax.NamedSharding(local_mesh, jax.P())

    params = jax.device_put(params, sharding)

    state = AsyncState(MRUparams=jax.device_get(params), updated=False)

    monitor_thread = threading.Thread(target=monitor_weight_sync, args=(state, async_config), daemon=True)
    monitor_thread.start()

    while not state.updated:
        print("Waiting for initial weights...")
        time.sleep(0.5)
    params = _maybe_update_params(params, state)

    i = 0
    while True:
        async_config.rollout_queue.put(f"rollout_{params.sum()}")
        time.sleep(0.2)
        i += 1
        params = _maybe_update_params(params, state)


def main():
    # TODO:
    # these would come from your config
    train_workers = 1
    inference_workers = 1
    jax.distributed.initialize()

    n = jax.device_count() // 2

    train_devices = np.array(jax.devices())[:n]
    inference_devices = np.array(jax.devices())[n:]

    train_mesh = jax.make_mesh((n,), ("dp",), axis_types=(jax.sharding.AxisType.Explicit,), devices=train_devices)
    inference_mesh = jax.make_mesh(
        (n,), ("dp",), axis_types=(jax.sharding.AxisType.Explicit,), devices=inference_devices
    )

    assert train_workers + inference_workers == jax.process_count(), (
        "Total number of workers must equal the number of JAX processes"
    )

    current_ip = get_current_vm_internal_ip()
    global_ip = get_global_ip()

    prompt_queue = queue.Queue()
    rollout_queue = queue.Queue()
    weight_sync_queue = queue.Queue(maxsize=inference_workers)

    QueueManager.register("get_prompt_queue", callable=lambda: prompt_queue)
    QueueManager.register("get_rollout_queue", callable=lambda: rollout_queue)
    QueueManager.register("get_weight_sync_queue", callable=lambda: weight_sync_queue)

    if current_ip == global_ip:
        start_server()

    sync_global_devices("serverReady")

    (prompt_queue_mp, rollout_queue_mp, weight_queue_mp) = get_queue(global_ip)

    async_config = AsyncConfig(
        train_workers=train_workers,
        inference_workers=inference_workers,
        prompt_queue=prompt_queue_mp,
        rollout_queue=rollout_queue_mp,
        weight_sync_queue=weight_queue_mp,
        train_mesh=train_mesh,
        inference_mesh=inference_mesh,
    )

    if jax.process_index() < train_workers:
        worker = TrainWorker(...)
        train_fn(async_config)
    else:
        inference_fn(async_config)

    print(f"Process at {current_ip} finished.")


if __name__ == "__main__":
    main()
