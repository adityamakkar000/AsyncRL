import os
from threading import Lock

import yaml
from dotenv import load_dotenv

load_dotenv()

CLUSTER_FILE = os.path.expanduser("~/.config/mesh/cluster.yaml")
DEFAULT_USER = os.environ.get("TPU_USERNAME", os.environ.get("USER", None))
IDENTITY_FILE = os.path.expanduser(os.environ.get("SSH_IDENTITY_FILE", "~/.ssh/id_rsa"))

lock = Lock()


def _load() -> dict:
    if not os.path.exists(CLUSTER_FILE):
        return {}
    with open(CLUSTER_FILE) as file:
        return yaml.safe_load(file) or {}


def _dump(config: dict) -> None:
    os.makedirs(os.path.dirname(CLUSTER_FILE), exist_ok=True)
    with open(CLUSTER_FILE, "w") as file:
        yaml.dump(config, file, sort_keys=False, default_flow_style=False)


def update_mesh_config(node_id, ip_list):
    """Updates or adds a node entry in the mesh cluster.yaml file."""
    with lock:
        full_config = _load()
        full_config[node_id] = {
            "user": DEFAULT_USER,
            "identity_file": IDENTITY_FILE,
            "hosts": ip_list,
        }
        _dump(full_config)


def delete_mesh_config(node_id):
    """Deletes a node entry from the mesh cluster.yaml file."""
    with lock:
        full_config = _load()
        if node_id in full_config:
            del full_config[node_id]
            _dump(full_config)
