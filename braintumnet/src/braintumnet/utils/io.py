import os, yaml, torch
from typing import Any, Dict

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_ckpt(model, path: str):
    ensure_dir(os.path.dirname(path))
    torch.save(model.state_dict(), path)

def load_ckpt(model, path: str, map_location="cpu"):
    sd = torch.load(path, map_location=map_location)
    model.load_state_dict(sd)
    return model
