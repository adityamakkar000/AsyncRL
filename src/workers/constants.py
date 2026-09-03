import socket

from jax._src.clusters.cloud_tpu_cluster import GceTpuCluster

PORT = 9000
KEY = b"aaa"
VM_IP = socket.gethostbyname(socket.gethostname())
GLOBAL_IP = GceTpuCluster.get_coordinator_address(60).split(":")[0]
TIMEOUT = 60 * 30  # 30 min
