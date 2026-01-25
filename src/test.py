import os

import stax
from jax.experimental.multihost_utils import broadcast_one_to_all
from stax.logger import staxLogger as logger

stax.init_distributed_jax()
logger.info(stax.utils.get_primary_host())
alpha = "abc999" + os.environ["RANK"]
out = broadcast_one_to_all(hash(alpha), is_source=(int(os.environ["RANK"]) == 0)).item()
print(out)