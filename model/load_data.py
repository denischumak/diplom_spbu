from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import numpy as np


def load_records(root_dir: str):
    root = Path(root_dir)
    records = []

    for meta_path in sorted(root.rglob("*_meta.json")):
        sample_dir = meta_path.parent
        raw_candidates = list(sample_dir.glob("*_raw.csv"))
        if not raw_candidates:
            continue

        raw_csv = raw_candidates[0]
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        gesture_id = meta.get("gesture_label", sample_dir.parent.parent.parent.name)
        subject_id = meta.get("subject_id", sample_dir.parent.parent.name)
        sampling_rate_hz = int(meta.get("sampling_rate_hz", 100))

        if not gesture_id or not subject_id:
            continue

        records.append(
            {
                "data": load_raw_csv(raw_csv).astype(np.float32, copy=False),
                "gesture_id": gesture_id,
                "subject_id": subject_id,
                "sampling_rate_hz": sampling_rate_hz,
            }
        )

    if not records:
        raise RuntimeError(f"No samples found under {root_dir}")

    return records


def split_records(records, leave_people: list, val_size=0.2, seed=42):
    _, others, strata = [], [], []
    for r in records:
        if r["subject_id"] not in leave_people:
            others.append(r)
            strata.append(f'{r["subject_id"]}__{r["gesture_id"]}')
    train, val = train_test_split(
        others,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
        stratify=strata,
    )

    return train, val


def load_raw_csv(raw_csv: Path) -> np.ndarray:
    """
    Returns shape [T, C].
    """
    arr = np.genfromtxt(raw_csv, delimiter=",", skip_header=1, dtype=np.float32)
    if arr.size == 0:
        raise ValueError(f"Empty CSV: {raw_csv}")
    if arr.ndim == 1:
        arr = arr[None, :]

    return arr.astype(np.float32)
