
import os
import json

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
