import model.utils as utils
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d


def augment_trimmed_sequence(sample: np.ndarray, augment_cfg, n_hall) -> np.ndarray:
    """
    sample is already trimmed and normalized.
    Augment only on trimmed segments.
    """
    data = sample.copy()
    T, C = data.shape

    # Имитирует постоянный сдвиг показаний с датчиков Холла
    if np.random.rand() < augment_cfg["p_offset"]:
        shift = np.random.uniform(
            -augment_cfg["hall_offset"],
            augment_cfg["hall_offset"],
            size=(n_hall,),
        )
        data[:, :n_hall] += shift

    # Имитирует микровибрации и шум АЦП
    if np.random.rand() < augment_cfg["p_jitter"]:
        # scale=0.03 означает шум с амплитудой 3% от стандартного отклонения
        noise = np.random.normal(
            loc=0.0, scale=augment_cfg["jitter_offset"], size=data.shape
        )
        data += noise


    # Случайно "отрезаем" края жеста и растягиваем обратно до длины T.
    if np.random.rand() < augment_cfg["p_warp"]:
       
        crop_ratio = np.random.uniform(1.0 - augment_cfg["max_crop_ratio"], 1.0)
        crop_len = int(T * crop_ratio)

       
        start_idx = np.random.randint(0, T - crop_len + 1)
        end_idx = start_idx + crop_len

        cropped_data = data[start_idx:end_idx, :]

        
        x_original = np.linspace(0, 1, crop_len)
        x_new = np.linspace(0, 1, T)

        interpolator = interp1d(x_original, cropped_data, axis=0, kind="linear")
        data = interpolator(x_new)

    return data.astype(np.float32)


def fit_normalizer_from_samples(samples):
    sum_ = None
    sumsq = None
    count = 0

    for s in samples:
        feat = utils.select_model_columns(s["data"])

        if feat.shape[0] == 0:
            continue

        if sum_ is None:
            sum_ = feat.sum(axis=0)
            sumsq = (feat**2).sum(axis=0)
        else:
            sum_ += feat.sum(axis=0)
            sumsq += (feat**2).sum(axis=0)

        count += feat.shape[0]

    if count == 0:
        raise RuntimeError("Could not fit normalizer: no samples after trimming")

    mean = sum_ / count
    var = sumsq / count - mean**2
    std = np.sqrt(np.maximum(var, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)


def estimate_target_len_from_samples(samples, quantile=0.95, min_target_len=128):
    lengths = [len(s["data"]) for s in samples if len(s["data"]) > 0]
    if not lengths:
        raise RuntimeError("Could not estimate target length")

    q = np.quantile(lengths, quantile)
    target_len = utils.round_up_to_multiple(q, 16)
    target_len = max(target_len, min_target_len)
    return int(target_len)


def get_rel_quat(sample):
    raw_quats = sample[:, 9:13]
    raw_quats = utils.fix_quaternion_flips(raw_quats)

    # [w, x, y, z] -> [x, y, z, w] for Scipy
    quats_scipy = np.roll(raw_quats, shift=-1, axis=1)

    q_current = R.from_quat(quats_scipy)
    q0_inv = q_current[0].inv()
    q_rel_obj = q0_inv * q_current
    res_scipy = q_rel_obj.as_quat()
    res_w_first = np.roll(res_scipy, shift=1, axis=1)

    sample[:, 9:13] = res_w_first

    return sample


def add_hall_diff_to_sample(sample, n_hall):
    """
    Собирает новый массив признаков в строгом порядке:
    [Halls, Hall_Diffs, Gyro, Acc, Quats]
    """
    halls = sample[:, :n_hall]
    hall_diffs = np.diff(halls, axis=0, prepend=halls[:1])
    others = sample[:, n_hall:]

   
    new_data = np.concatenate([halls, hall_diffs, others], axis=1)

    return new_data


def preprocess_data(records, n_hall, add_hall_diff, trimmer=None):
    """Trims all samples, adds hall_diff channels"""
    samples = []
    for rec in records:
        sample = rec["data"]
        if trimmer is not None:
            trim_in = utils.select_trim_columns(sample)
            start, end = trimmer.trim(trim_in)
            sample = sample[start:end]
        if sample.shape[0] < 10:
            continue
        sample = get_rel_quat(sample)
        if add_hall_diff:
            sample = add_hall_diff_to_sample(sample, n_hall)
        samples.append(
            {
                "data": sample,
                "gesture_id": rec["gesture_id"],
                "subject_id": rec["subject_id"],
            }
        )

    return samples
