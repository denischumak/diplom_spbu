# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))



from load_data import load_records
import config as cfg

records = [r for r in load_records(cfg.DATA_ROOT)]
info = {}
labels = sorted({r["gesture_id"] for r in records})
for r in records:
    if r["subject_id"] not in info:
        info[r["subject_id"]] = {l: 0 for l in labels}
    info[r["subject_id"]][r["gesture_id"]] += 1
print(info)
