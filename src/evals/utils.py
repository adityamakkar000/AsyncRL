import subprocess
import requests
from loguru import logger
import json
import hashlib
from omegaconf import DictConfig, OmegaConf


def ping_server(host: str, port: int) -> bool:
    try:
        response = requests.get(f"http://{host}:{port}/ping", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def terminate_process(process: subprocess.Popen | None, name: str) -> None:
    if process is not None:
        logger.info(f"Terminating {name} process...")
        # process might have already exited
        try:
            process.terminate()
            process.wait()
        except Exception as e:
            ...
        logger.info(f"{name} process terminated.")
    return None


def format_command(cmd: list[str]) -> str:
    out = [cmd[0] + " " + cmd[1]]
    i = 2
    while i < len(cmd):
        if cmd[i].startswith("--"):
            if i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
                out.append(f"\t{cmd[i]} {cmd[i + 1]}")
                i += 2
            else:
                out.append(f"\t{cmd[i]}")
                i += 1
        else:
            out.append(f"\t{cmd[i]}")
            i += 1
    return "\n".join(out)


def hash_dictConfig(d: DictConfig) -> str:
    hash_obj = OmegaConf.to_container(
        d,
        resolve=True,
        throw_on_missing=True,
    )
    hash_dict = json.dumps(hash_obj, sort_keys=True)
    return hashlib.sha256(hash_dict.encode("utf-8")).hexdigest()
