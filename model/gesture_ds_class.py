import model.utils as utils
import torch
from torch.utils.data import Dataset
import model.preprocessing as preprocessing
import numpy as np


class GestureDataset(Dataset):
    def __init__(
        self,
        samples,
        label2idx,
        mean,
        std,
        target_len,
        n_hall,
        train=False,
        augment=False,
        augment_cfg=None,
    ):
        if augment_cfg is None and augment:
            raise ValueError("If augment=True, augment_cfg can't be None")
        self.samples = samples
        self.label2idx = label2idx
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-3)
        self.target_len = int(target_len)
        self.train = train
        self.augment = augment
        self.augment_cfg = augment_cfg
        self.n_hall = n_hall

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        feat = utils.select_model_columns(s["data"])
        feat = (feat - self.mean) / self.std

        if self.train and self.augment:
            feat = preprocessing.augment_trimmed_sequence(
                feat, self.augment_cfg, self.n_hall
            )

        t, c = feat.shape
        if t > self.target_len:
            left = (t - self.target_len) // 2
            feat = feat[left : left + self.target_len]
            valid_len = self.target_len
        else:
            valid_len = t
            out = np.zeros((self.target_len, c), dtype=np.float32)
            out[:t] = feat  # zero-padding in the end
            feat = out

        y = self.label2idx[s["gesture_id"]]

        x_tensor = torch.from_numpy(feat.T).float()  # [C, T]
        len_tensor = torch.tensor(valid_len, dtype=torch.long)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return x_tensor, len_tensor, y_tensor, s["subject_id"]
