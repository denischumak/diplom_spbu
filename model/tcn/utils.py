import numpy as np
import random
import torch
import math


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def round_up_to_multiple(x: float, m: int = 16) -> int:
    return int(math.ceil(float(x) / m) * m)


def select_trim_columns(x: np.ndarray) -> np.ndarray:
    """
    return all sensors
    """
    return x


def select_model_columns(x: np.ndarray) -> np.ndarray:
    """
    return all sensors
    """
    return x


def fix_quaternion_flips(quats):
    """
    Убирает резкие скачки знака (q -> -q).
    quats: массив формы (N, 4) в формате [x, y, z, w] или [w, x, y, z]
    """
    for i in range(1, len(quats)):
        dot_product = np.dot(quats[i], quats[i - 1])
        if dot_product < 0:  # flip detected
            quats[i] = -quats[i]
    return quats

def compute_person_weights(train_records, person2id):
    subject_labels = np.array([person2id[r["subject_id"]] for r in train_records], dtype=np.int64)
    counts = np.bincount(subject_labels, minlength=len(person2id))
    weights = 1.0 / (np.maximum(counts, 1))
    sample_weights = [weights[subject_id] for subject_id in subject_labels]
    return torch.tensor(sample_weights, dtype=torch.float32)


