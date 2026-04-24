import json
from pathlib import Path
import numpy as np


def load_trim_dataset(root_dir: str):
    root = Path(root_dir)
    dataset = []

    gt_files = list(root.rglob("*_gt.json"))
    for gt_path in gt_files:
        sample_dir = gt_path.parent

        raw_candidates = list(sample_dir.glob("*_raw.csv"))
        meta_candidates = list(sample_dir.glob("*_meta.json"))

        if not raw_candidates or not meta_candidates:
            print(f"[WARN] missing raw/meta in {sample_dir}")
            continue

        raw_csv = raw_candidates[0]
        meta_json = meta_candidates[0]

        # load raw csv
        data = np.genfromtxt(raw_csv, delimiter=",", skip_header=1, dtype=np.float32)

        if data.size == 0:
            print(f"[WARN] empty raw csv: {raw_csv}")
            continue
        if data.ndim == 1:
            data = data[None, :]

        
        # load meta
        with meta_json.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        # load gt
        with gt_path.open("r", encoding="utf-8") as f:
            gt = json.load(f)

        gt_start = int(gt["gt_start"])
        gt_end = int(gt["gt_end"])
        fs = int(meta["sampling_rate_hz"])

        # sanity check
        if not (0 <= gt_start < gt_end < len(data)):
            print(
                f"[WARN] bad gt range in {gt_path}: "
                f"gt_start={gt_start}, gt_end={gt_end}, len={len(data)}"
            )
            continue

        dataset.append(
            {
                "data": data,
                "gt_start": gt_start,
                "gt_end": gt_end,
                "sampling_rate_hz": fs,
                "gesture_label": meta.get("gesture_label", sample_dir.parent.parent.name if sample_dir.parent.parent else ""),
                "subject_id": meta.get("subject_id", sample_dir.parent.name if sample_dir.parent else ""),
            }
        )

    print(f"[INFO] loaded {len(dataset)} samples from {root}")
    return dataset
