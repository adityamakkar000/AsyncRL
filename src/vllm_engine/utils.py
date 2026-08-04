import subprocess

import requests
from loguru import logger


def ping_server(host: str, port: int | str) -> bool:
    try:
        response = requests.get(f"http://{host}:{port}/ping", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


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


def terminate_process(process: subprocess.Popen | None, name: str) -> None:
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
