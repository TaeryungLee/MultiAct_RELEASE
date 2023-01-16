import os
import sys


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def add_to_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
