import os

import yaml

CLUSTER_FILE = os.path.expanduser("~/.config/mesh/cluster.yaml")
DEFAULT_USER = os.environ.get("USER")
IDENTITY_FILE = os.path.expanduser("~/.ssh/id_rsa")


def update_mesh_config(node_id, ip_list):
    """Updates or adds a node entry in the mesh cluster.yaml file."""
    with open(CLUSTER_FILE, "r") as file:
        full_config = yaml.safe_load(file)

    full_config[node_id] = {
        "user": DEFAULT_USER,
        "identity_file": IDENTITY_FILE,
        "hosts": ip_list,
    }

    with open(CLUSTER_FILE, "w") as file:
        yaml.dump(full_config, file, sort_keys=False, default_flow_style=False)
    print(f"[Config] cluster.yaml updated for {node_id}")
