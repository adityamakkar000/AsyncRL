import queue
import threading
import time

import hydra
import jax
import numpy as np
import stax
from hydra.core.config_store import ConfigStore
from jax.experimental.multihost_utils import sync_global_devices
from jax.sharding import AxisType
from omegaconf import DictConfig, OmegaConf
from stax.logger import staxLogger as logger

from src.constants import GLOBAL_IP, KEY, PORT, VM_IP, AsyncOptions, QueueManager
from src.workers import AsyncInferenceWorker, AsyncTrainerWorker, TrainerConfig

cs = ConfigStore.instance()
cs.store(name="base", node=TrainerConfig)


def start_server():
    def _start_server():
        manager = QueueManager(address=("0.0.0.0", PORT), authkey=KEY)
        server = manager.get_server()
        logger.info(f"[Server] Queue server listening on {GLOBAL_IP}:{PORT}...")
        server.serve_forever()

    server_thread = threading.Thread(target=_start_server, daemon=True)
    server_thread.start()


def get_queues():
    manager = QueueManager(address=(GLOBAL_IP, PORT), authkey=KEY)

    RETRIES = 6
    for _ in range(RETRIES):
        try:
            manager.connect()
            return (
                manager.get_prompt_queue(),  # type: ignore
                manager.get_rollout_queue(),  # type: ignore
                manager.get_weight_sync_queue(),  # type: ignore
                manager.get_inference_metrics_queue(),  # type: ignore
            )
        except ConnectionError:
            logger.info(f"[Client] Waiting for server at {GLOBAL_IP}...", log_for_all=True)
            time.sleep(1)

    raise ConnectionError(f"Could not connect to server at {GLOBAL_IP} after multiple attempts.")


@hydra.main(version_base=None, config_path="./configs/train")
def main(cfg: DictConfig) -> None:
    stax.init_distributed_jax()

    train_workers = cfg.async_config.train_workers
    assert (n_hosts := jax.process_count()) > train_workers > 0, (
        "Number of train workers must be between 1 and total number of processes - 1"
    )
    assert n_hosts % train_workers == 0, (
        f"Number of train workers must divide total processes, got {n_hosts} and {train_workers}"
    )
    inference_workers = n_hosts - train_workers

    devices = np.array(jax.devices())
    devices_per_host = jax.local_device_count()

    train_devices = devices[: train_workers * devices_per_host].reshape(train_workers, devices_per_host)
    inference_devices = devices[train_workers * devices_per_host :].reshape(inference_workers, devices_per_host)

    train_mesh = jax.sharding.Mesh(
        train_devices,
        ("processes", "local_devices"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )

    inference_mesh = jax.sharding.Mesh(
        inference_devices,
        ("processes", "local_devices"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    global_mesh = jax.make_mesh(
        (jax.process_count(), jax.local_device_count()),
        ("processes", "local_devices"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )

    local_prompt_queue = queue.Queue(
        maxsize=(cfg.data_config.batch_size // cfg.loss_config.inference_config.group_size)
    )
    local_rollout_queue = queue.Queue()
    local_weight_sync_queue = queue.Queue(maxsize=inference_workers)
    local_inference_metrics_queue = queue.Queue()

    QueueManager.register("get_prompt_queue", callable=lambda: local_prompt_queue)
    QueueManager.register("get_rollout_queue", callable=lambda: local_rollout_queue)
    QueueManager.register("get_weight_sync_queue", callable=lambda: local_weight_sync_queue)
    QueueManager.register("get_inference_metrics_queue", callable=lambda: local_inference_metrics_queue)

    if VM_IP == GLOBAL_IP:
        start_server()

    sync_global_devices("serverReady")

    global_prompt_queue, global_rollout_queue, global_weight_sync_queue, global_inference_metrics_queue = get_queues()

    async_options = AsyncOptions(
        train_workers=train_workers,
        inference_workers=inference_workers,
        prompt_queue=global_prompt_queue,
        rollout_queue=global_rollout_queue,
        weight_sync_queue=global_weight_sync_queue,
        inference_metrics_queue=global_inference_metrics_queue,
        train_mesh=train_mesh,
        inference_mesh=inference_mesh,
        global_mesh=global_mesh,
    )

    logger.info(OmegaConf.to_yaml(cfg), log_for_all=True)

    rank = stax.get_rank()
    worker = (AsyncTrainerWorker if rank < train_workers else AsyncInferenceWorker)(cfg, async_options)  # type: ignore
    worker.start()

    logger.info(f"Process at {VM_IP} finished.", log_for_all=True)


if __name__ == "__main__":
    main()
