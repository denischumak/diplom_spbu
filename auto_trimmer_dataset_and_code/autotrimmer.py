# import numpy as np


# class AutoTrimmer:
#     def __init__(self, cfg):
#         self.cfg = cfg

#     def moving_average(self, x, w):
#         if w <= 1:
#             return x
#         return np.convolve(x, np.ones(w) / w, mode="same")

#     def _compute_score(self, data, sens, compute_for):
#         """data: (T, 3) для одного датчика"""
#         if sens not in ["gyro", "acc", "quat"]:
#             raise ValueError("Unknown sensor type for score computation")
#         if compute_for == "end":
#             data = data[::-1]  # reverse for end boundary detection
#         cfg = self.cfg
#         baseline = data[: cfg[f"baseline_len_{compute_for}"]].mean(axis=0)
#         if sens == "quat":
#             q0 = baseline / np.linalg.norm(baseline)
#             dots = np.abs(np.dot(data, q0))  # (T,)
#             dots = np.clip(dots, 0.0, 1.0)
#             angle = np.rad2deg(2 * np.arccos(dots))
#             base_angle = angle[: cfg[f"baseline_len_{compute_for}"]]
#             base_std = max(base_angle.std(), 1e-3)
#             score = angle / base_std
#         else:
#             centered = data - baseline
#             base_std = data[: cfg[f"baseline_len_{compute_for}"]].std(axis=0)
#             scale = np.maximum(base_std, 1e-3)
#             score = np.abs(centered / scale)  # (T,3)
#             score = np.mean(score, axis=1)  # L1 по осям
#         score = self.moving_average(score, cfg[f"smooth_win_{sens}"])
#         return score / np.max(score)

#     def find_boundary(self, score, cfg, compute_for):
#         boundary = None
#         for i in range(cfg[f"baseline_len_{compute_for}"], len(score)):
#             if np.all(score[i : i + cfg[f"min_{compute_for}_run"]] > cfg[f"thr_{compute_for}"]):
#                 boundary = i
#                 break
#         if boundary is None:
#             boundary = 0 if compute_for == "start" else len(score)
#         return boundary

#     def get_quat_angles(self, quat):
#         cfg = self.cfg
#         baseline = quat[: cfg[f"baseline_len_start"]].mean(axis=0)
#         q0 = baseline / np.linalg.norm(baseline)
#         dots = np.abs(np.dot(quat, q0))  # (T,)
#         dots = np.clip(dots, 0.0, 1.0)
#         angle = np.rad2deg(2 * np.arccos(dots))
#         return angle

#     def trim(self, x):
#         """data: (T, 13)"""
#         t, axs = x.shape
#         if axs != 13:
#             raise ValueError(
#                 "Number of channels must be set to 13 (x[0:3] - hall, x[3:6] - gyro, x[6:9] - accel, x[9:13] - quats)"
#             )
#         cfg = self.cfg

#         score_gyro_start = self._compute_score(
#             x[:, 3:6], sens="gyro", compute_for="start"
#         )
#         score_gyro_end = self._compute_score(
#             x[:, 3:6], sens="gyro", compute_for="end"
#         )

#         score_acc_start = self._compute_score(
#             x[:, 6:9], sens="acc", compute_for="start"
#         )
#         score_acc_end = self._compute_score(
#             x[:, 6:9], sens="acc", compute_for="end"
#         )

#         score_quat_start = self._compute_score(
#             x[:, 9:13], sens="quat", compute_for="start"
#         )
#         score_quat_end = self._compute_score(
#             x[:, 9:13], sens="quat", compute_for="end"
#         )

#         score_start = (
#             self.cfg["gyro_weight"] * score_gyro_start
#             + self.cfg["acc_weight"] * score_acc_start
#             + self.cfg["quat_weight"] * score_quat_start
#         )

#         score_end = (
#             self.cfg["gyro_weight"] * score_gyro_end
#             + self.cfg["acc_weight"] * score_acc_end
#             + self.cfg["quat_weight"] * score_quat_end
#         )

#         # find start and end boundaries
#         start = self.find_boundary(score_start, cfg, "start")
#         end = t - self.find_boundary(score_end, cfg, "end")

#         start = max(0, start - cfg["margin_start"])
#         end = min(t, end + cfg["margin_end"])
#         # если был выброс, значения threshold могут быть слишком завышенными (большой пик - меньше порог)
#         # 1000 (в конце 30): [0.06, 0.03, 0.03, 0.03, 0.03], 500 (в конце 30): [0.12, 0.06,0.06,0.06,0.06], thr = 0.1
#         return start, end


import numpy as np


class AutoTrimmer:
    def __init__(self, cfg):
        self.cfg = cfg

    def moving_average(self, x, w):
        if w <= 1:
            return x
        return np.convolve(x, np.ones(w) / w, mode="same")

    def _compute_score(self, data, sens, compute_for):
        """data: (T, 3) для одного датчика"""
        if sens not in ["gyro", "acc", "quat"]:
            raise ValueError("Unknown sensor type for score computation")
        if compute_for == "end":
            data = data[::-1]  # reverse for end boundary detection
        cfg = self.cfg
        baseline = data[: cfg[f"baseline_len_{compute_for}"]].mean(axis=0)
        if sens == "quat":
            q0 = baseline / np.linalg.norm(baseline)
            dots = np.abs(np.dot(data, q0))  # (T,)
            dots = np.clip(dots, 0.0, 1.0)
            angle = np.rad2deg(2 * np.arccos(dots))
            base_angle = angle[: cfg[f"baseline_len_{compute_for}"]]
            base_std = max(base_angle.std(), 1e-3)
            score = angle / base_std
        else:
            centered = data - baseline
            base_std = data[: cfg[f"baseline_len_{compute_for}"]].std(axis=0)
            scale = np.maximum(base_std, 1e-3)
            score = np.abs(centered / scale)  # (T,3)
            score = np.mean(score, axis=1)  # L1 по осям
        score = self.moving_average(score, cfg[f"smooth_win_{sens}"])
        peak = np.percentile(score, 99)
        if peak > 1e-4:
            score = score / peak
        else:
            score = score * 0.0
        return np.clip(score, 0.0, 1.0)  # ADD PERCENTILE

    def find_boundary(self, score, cfg, compute_for):
        boundary = None
        for i in range(cfg[f"baseline_len_{compute_for}"], len(score)):
            if np.all(
                score[i : i + cfg[f"min_{compute_for}_run"]] > cfg[f"thr_{compute_for}"]
            ):
                boundary = i
                break
        if boundary is None:
            boundary = 0 if compute_for == "start" else len(score)
        return boundary

    def get_quat_angles(self, quat):
        cfg = self.cfg
        baseline = quat[: cfg[f"baseline_len_start"]].mean(axis=0)
        q0 = baseline / np.linalg.norm(baseline)
        dots = np.abs(np.dot(quat, q0))  # (T,)
        dots = np.clip(dots, 0.0, 1.0)
        angle = np.rad2deg(2 * np.arccos(dots))
        return angle

    def trim(self, x):
        """data: (T, 13)"""
        t, axs = x.shape
        if axs != 13:
            raise ValueError(
                "Number of channels must be set to 13 (x[0:3] - hall, x[3:6] - gyro, x[6:9] - accel, x[9:13] - quats)"
            )
        cfg = self.cfg

        score_gyro_start = self._compute_score(
            x[:, 3:6], sens="gyro", compute_for="start"
        )
        score_gyro_end = self._compute_score(x[:, 3:6], sens="gyro", compute_for="end")

        score_acc_start = self._compute_score(
            x[:, 6:9], sens="acc", compute_for="start"
        )
        score_acc_end = self._compute_score(x[:, 6:9], sens="acc", compute_for="end")

        score_quat_start = self._compute_score(
            x[:, 9:13], sens="quat", compute_for="start"
        )
        score_quat_end = self._compute_score(x[:, 9:13], sens="quat", compute_for="end")

        score_start = (
            self.cfg["gyro_weight"] * score_gyro_start
            + self.cfg["acc_weight"] * score_acc_start
            + self.cfg["quat_weight"] * score_quat_start
        )

        score_end = (
            self.cfg["gyro_weight"] * score_gyro_end
            + self.cfg["acc_weight"] * score_acc_end
            + self.cfg["quat_weight"] * score_quat_end
        )

        # find start and end boundaries
        # score_start /= np.max(score_start)
        # score_end /= np.max(score_end)
        start = self.find_boundary(score_start, cfg, "start")
        end = t - self.find_boundary(score_end, cfg, "end")

        start = max(0, start - cfg["margin_start"])
        end = min(t, end + cfg["margin_end"])
        # если был выброс, значения threshold могут быть слишком завышенными (большой пик - меньше порог)
        # 1000 (в конце 30): [0.06, 0.03, 0.03, 0.03, 0.03], 500 (в конце 30): [0.12, 0.06,0.06,0.06,0.06], thr = 0.1
        return start, end
