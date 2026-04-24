import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

class AutoTrimmer:
    def __init__(self, trim_cfg, sens_cfg):
        self.trim_cfg = trim_cfg
        self.sens_cfg = sens_cfg

    def moving_average(self, x, w):
        if w <= 1:
            return x
        return np.convolve(x, np.ones(w) / w, mode="same")

    def _compute_score(self, data, sens, compute_for):
        """
        data: (T, 3) для одного датчика.
        Для compute_for='start' data должна быть в прямом порядке,
        для compute_for='end' data должна быть уже перевёрнута.
        """
        if sens not in ["gyro", "acc", "quat"]:
            raise ValueError("Unknown sensor type for score computation")
        trim_cfg = self.trim_cfg
        baseline = data[: trim_cfg[f"baseline_len_{compute_for}"]].mean(axis=0)
        if sens == "quat":
            q0 = baseline / np.linalg.norm(baseline)
            dots = np.abs(np.dot(data, q0))  # (T,)
            dots = np.clip(dots, 0.0, 1.0)
            angle = np.rad2deg(2 * np.arccos(dots))
            base_angle = angle[: trim_cfg[f"baseline_len_{compute_for}"]]
            base_std = max(base_angle.std(), 1e-3)
            score = angle / base_std
        else:
            centered = data - baseline
            base_std = data[: trim_cfg[f"baseline_len_{compute_for}"]].std(axis=0)
            scale = np.maximum(base_std, 1e-3)
            score = np.abs(centered / scale)  # (T,3)
            score = np.mean(score, axis=1)  # L1 по осям

        score = self.moving_average(score, trim_cfg[f"smooth_win_{sens}"])
        peak = np.percentile(score, 99)

        if peak > 1e-4:
            return np.clip(score / peak, 0.0, 1.0)
        return score * 0.0

    def _find_boundary(self, score, compute_for):
        trim_cfg = self.trim_cfg
        baseline_len = trim_cfg[f"baseline_len_{compute_for}"]
        min_run = trim_cfg[f"min_{compute_for}_run"]
        for i in range(baseline_len, len(score) - min_run + 1):
            if np.all(score[i : i + min_run] > trim_cfg[f"thr_{compute_for}"]):
                return i
        result = 0 if compute_for == "start" else len(score)
        return result

    def get_quat_angles(self, quat):
        trim_cfg = self.trim_cfg
        baseline = quat[: trim_cfg[f"baseline_len_start"]].mean(axis=0)
        q0 = baseline / np.linalg.norm(baseline)
        dots = np.dot(quat, q0)  # (T,)
        dots = np.clip(dots, -1.0, 1.0)
        angle = np.rad2deg(2 * np.arccos(dots))
        return angle

    def trim(self, x):
        """Returns start (inclusive), end (exclusive) for record"""
        t, axs = x.shape
        if axs != 13:
            raise ValueError("Expected 13 channels")
        trim_cfg = self.trim_cfg
        sens_cfg = self.sens_cfg

        # Выделяем каналы
        n_hall_acc = sens_cfg["n_hall"] + sens_cfg["n_acc"]
        acc = x[:, sens_cfg["n_hall"] : n_hall_acc]
        gyro = x[:, n_hall_acc : n_hall_acc + sens_cfg["n_gyro"]]
        quat = x[:, n_hall_acc + sens_cfg["n_gyro"] : ]

        # Реверсированные копии (для поиска конца)
        acc_rev = acc[::-1]
        gyro_rev = gyro[::-1]
        quat_rev = quat[::-1]

        # Скоры для старта (на прямых данных)
        s_acc_s = self._compute_score(acc, "acc", "start")
        s_gyro_s = self._compute_score(gyro, "gyro", "start")
        s_quat_s = self._compute_score(quat, "quat", "start")

        # Скоры для конца (на перевёрнутых данных)
        s_acc_e = self._compute_score(acc_rev, "acc", "end")
        s_gyro_e = self._compute_score(gyro_rev, "gyro", "end")
        s_quat_e = self._compute_score(quat_rev, "quat", "end")

        # Взвешенная сумма
        score_start = (
            trim_cfg["gyro_weight"] * s_gyro_s
            + trim_cfg["acc_weight"] * s_acc_s
            + trim_cfg["quat_weight"] * s_quat_s
        )
        score_end = (
            trim_cfg["gyro_weight"] * s_gyro_e
            + trim_cfg["acc_weight"] * s_acc_e
            + trim_cfg["quat_weight"] * s_quat_e
        )

        # Границы
        start = self._find_boundary(score_start, "start")
        end = t - self._find_boundary(score_end, "end") 

        # Отступы
        start = max(0, start - trim_cfg["margin_start"])
        end = min(t, end + trim_cfg["margin_end"])
        if not start < end:
            return 0, t
        
        return start, end
