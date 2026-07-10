from pathlib import Path
import json


def load_trim_cfg(path: str):
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg
