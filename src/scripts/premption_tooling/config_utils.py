import os
from threading import Lock

import yaml
from dotenv import load_dotenv

load_dotenv()

cluster_file = os.path.expanduser("~/.config/mesh/cluster.yaml")
default_user = os.environ.get("USER")
identity_file = os.path.expanduser(os.environ.get("SSH_IDENTITY_FILE", "~/.ssh/id_rsa"))

lock = Lock()


def update_mesh_config(node_id, ip_list):
    """Updates or adds a node entry in the mesh cluster.yaml file."""
    with lock:
        with open(cluster_file, "r") as file:
            full_config = yaml.safe_load(file)

        full_config[node_id] = {
            "user": default_user,
            "identity_file": identity_file,
            "hosts": ip_list,
        }

        with open(cluster_file, "w") as file:
            yaml.dump(full_config, file, sort_keys=False, default_flow_style=False)


def delete_mesh_config(node_id):
    """Deletes a node entry from the mesh cluster.yaml file."""
    with lock:
        with open(cluster_file, "r") as file:
            full_config = yaml.safe_load(file)

        if node_id in full_config:
            del full_config[node_id]

        with open(cluster_file, "w") as file:
            yaml.dump(full_config, file, sort_keys=False, default_flow_style=False)
