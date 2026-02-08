# print(__name__)

# debug_paths.py
import os, sys
from pathlib import Path

print("=== identity ===")
print("__name__:", __name__)
print("__package__:", __package__)
print("__file__:", __file__)

print("\n=== where you ran it from ===")
print("cwd:", os.getcwd())
print("argv[0]:", sys.argv[0])
print("script path:", Path(__file__).resolve())
print("script dir :", Path(__file__).resolve().parent)

print("\n=== sys.path (import search paths) ===")
for i, p in enumerate(sys.path):
    print(f"{i:02d}: {p}")

print("\n=== env vars that often affect imports ===")
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))

print("\n=== key interpretation ===")
print("sys.path[0]:", sys.path[0], "(often script dir; '' means cwd)")
