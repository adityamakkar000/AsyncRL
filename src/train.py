import hydra
import jax
import numpy as np
import stax
from hydra.core.config_store import ConfigStore
from jax.experimental.multihost_utils import sync_global_devices
from jax.sharding import AxisType
from omegaconf import DictConfig, OmegaConf
from stax.logger import staxLogger as logger

from src.workers import AsyncInferenceWorker, AsyncTrainerWorker, TrainerConfig
from src.workers.config import AsyncOptions
from src.workers.constants import GLOBAL_IP, VM_IP
from src.workers.utils import MPQueues

cs = ConfigStore.instance()
cs.store(name="base", node=TrainerConfig)


@hydra.main(version_base=None, config_path="./configs/train")
def main(cfg: DictConfig) -> None:
    stax.init_distributed_jax()

    train_workers = cfg.async_config.train_workers
    n_hosts = jax.process_count()
    assert n_hosts > train_workers > 0, "Number of train workers must be between 1 and total number of processes - 1"
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

    queues = MPQueues(GLOBAL_IP)
    queues.register(
        maxsizes={
            "prompt_queue": cfg.loss_config.inference_config.max_decode_batch_size * jax.device_count(),
            "weight_sync_queue": inference_workers,
        }
    )

    if VM_IP == GLOBAL_IP:
        queues.start_server()
    sync_global_devices("serverReady")

    queues.connect()

    async_options = AsyncOptions(
        train_workers=train_workers,
        inference_workers=inference_workers,
        queues=queues,
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
