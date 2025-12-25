import subprocess
import requests
from loguru import logger
import json
import hashlib
from omegaconf import DictConfig, OmegaConf


def ping_server(host: str, port: int | str) -> bool:
    """
    Ping a server to check if it's alive.
    Args:
        host (str): The server host.
        port (int): The server port.
    Returns:
        bool: True if the server responds, False otherwise.
    """
    try:
        response = requests.get(f"http://{host}:{port}/ping", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def terminate_process(process: subprocess.Popen | None, name: str) -> None:
    """
    Terminate a subprocess if it exists.
    Args:
        process (subprocess.Popen | None): The process to terminate.
        name (str): Name of the process for logging purposes.
    Returns:
        None

    """
    if process is not None:
        logger.info(f"Terminating {name} process...")
        # process might have already exited
        try:
            process.terminate()
            process.wait()
        except Exception:
            ...
        logger.info(f"{name} process terminated.")
    return None


def format_command(cmd: list[str]) -> str:
    """
    Format a command list into a readable string.
    Args:
        cmd (list[str]): The command as a list of strings.
    Returns:
        str: Formatted command string.
    """

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
    """
    Hash a DictConfig object.
    Args:
        d (DictConfig): The DictConfig object to hash.
    Returns:
        str: The SHA-256 hash of the DictConfig.
    """

    # TODO:
    # find a way to only include some keys

    hash_obj = OmegaConf.to_container(
        d,
        resolve=True,
        throw_on_missing=True,
    )
    hash_dict = json.dumps(hash_obj, sort_keys=True)
    return hashlib.sha256(hash_dict.encode("utf-8")).hexdigest()
