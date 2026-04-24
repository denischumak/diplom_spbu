import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import load_data
import config

records = load_data.load_records(config.DATA_ROOT)
labels = sorted({r["gesture_id"] for r in records})
subject_ids = sorted({r["subject_id"] for r in records})
results = {}
for r in records:
    subject = r["subject_id"]
    if subject not in results:
        results[subject] = {l: 0 for l in labels}
    results[subject][r["gesture_id"]] += 1
print(results)

