import argparse
import time

from .sandbox import Sandbox, SandboxConfig

parser = argparse.ArgumentParser(description="Start a sandbox, run a command in it, then destroy it.")
parser.add_argument("command", nargs="?", default="python -c 'print(sum(range(10)))'")
parser.add_argument("--runtime", default="runsc", help="'runsc' for gVisor, 'runc' for plain docker")
parser.add_argument("--timeout", type=float, default=30.0)
args = parser.parse_args()

config = SandboxConfig(runtime=None if args.runtime == "runc" else args.runtime)
start = time.perf_counter()
with Sandbox(config) as sandbox:
    print(f"started {sandbox.container.name} in {time.perf_counter() - start:.2f}s")
    print(sandbox.exec(args.command, timeout=args.timeout))
