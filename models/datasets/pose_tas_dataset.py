import os
import numpy as np
import torch
from torch.utils.data import Dataset

class PoseTASDataset(Dataset):
    def __init__(self, list_path, feature_dir, label_dir, max_frames=None):
        with open(list_path, "r") as f:
            self.sample_list = [line.strip() for line in f]

        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.max_frames = max_frames

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample_id = self.sample_list[idx]

        x = np.load(os.path.join(self.feature_dir, sample_id + ".npy"))  # shape (T, 51)
        y = np.load(os.path.join(self.label_dir, sample_id + ".npy"))    # shape (T,)

        T = x.shape[0]

        if self.max_frames:
            if T > self.max_frames:
                x = x[:self.max_frames]
                y = y[:self.max_frames]
            else:
                pad_len = self.max_frames - T
                x = np.pad(x, ((0, pad_len), (0, 0)), constant_values=0)
                y = np.pad(y, (0, pad_len), constant_values=-100)  # -100 is ignored by CrossEntropyLoss

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
