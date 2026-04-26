import os
from threading import Lock

import yaml
from dotenv import load_dotenv

load_dotenv()

CLUSTER_FILE = os.path.expanduser("~/.config/mesh/cluster.yaml")
DEFAULT_USER = os.environ.get("TPU_USERNAME")
IDENTITY_FILE = os.path.expanduser(os.environ.get("SSH_IDENTITY_FILE", "~/.ssh/id_rsa"))

lock = Lock()


def update_mesh_config(node_id, ip_list):
    """Updates or adds a node entry in the mesh cluster.yaml file."""
    with lock:
        with open(CLUSTER_FILE, "r") as file:
            full_config = yaml.safe_load(file)

        full_config[node_id] = {
            "user": DEFAULT_USER,
            "identity_file": IDENTITY_FILE,
            "hosts": ip_list,
        }

        with open(CLUSTER_FILE, "w") as file:
            yaml.dump(full_config, file, sort_keys=False, default_flow_style=False)


def delete_mesh_config(node_id):
    """Deletes a node entry from the mesh cluster.yaml file."""
    with lock:
        with open(CLUSTER_FILE, "r") as file:
            full_config = yaml.safe_load(file)

        if node_id in full_config:
            del full_config[node_id]

        with open(CLUSTER_FILE, "w") as file:
            yaml.dump(full_config, file, sort_keys=False, default_flow_style=False)
