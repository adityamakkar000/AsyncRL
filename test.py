import argparse

import jax
import jax._src.distributed as dist
from orbax.checkpoint._src.multihost import multihost as orbax_multihost


def parse_active(s: str):
    if s == "all":
        return None
    else:
        return {int(x) for x in s.split(",") if x.strip()}


prefix = "stax_active_processes/"
published = False
cache = dict()


def _publish_runtime_to_distributed():
    global published
    if published:
        return
    own_rt = jax.process_index()
    own_dist = jax._src.distributed.global_state.process_id
    client = jax._src.distributed.global_state.client
    client.key_value_set(f"{prefix}{own_rt}", str(own_dist), allow_overwrite=True)
    cache[own_rt] = own_dist
    published = True


def lookup_runtime_to_distributed(rt: int):
    if rt in cache:
        return cache[rt]
    client = jax._src.distributed.global_state.client
    dist_id = client.blocking_key_value_get(f"{prefix}{rt}", 100)
    cache[rt] = int(dist_id)
    return cache[rt]


def apply_client_fix():
    orbax_multihost.use_experimental_distributed_process_id = lambda: True
    _publish_runtime_to_distributed()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--active", type=str, default="all")
    args = ap.parse_args()

    processes = parse_active(args.active)

    args = ap.parse_args()
    active = parse_active(args.active)
    jax.distributed.initialize()

    apply_client_fix()
    runtime = jax.process_index()
    distrbuted_id = dist.global_state.process_id

    print(f"Hello from process {runtime} (distributed id: {distrbuted_id})")
    if active is None or runtime in active:
        if active is not None:
            processes = {lookup_runtime_to_distributed(rt) for rt in processes}

        print(f"[pid={runtime}] I am active!")
        orbax_multihost.sync_global_processes("test_sync", processes=processes)
        print(f"[pid={runtime}] Finished sync!")
    else:
        print(f"[pid={runtime}] I am inactive, skipping sync.")


if __name__ == "__main__":
    main()
