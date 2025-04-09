import os
import numpy as np
import torch
from yacs.config import CfgNode
from models.datasets.pose_tas_dataset import PoseTASDataset


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        self.index = 0
        self.num_batch = int(np.ceil(len(dataset) / batch_size))

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_batch

    def __next__(self):
        if self.index >= len(self.dataset):
            if self.shuffle:
                np.random.shuffle(self.indices)
            self.index = 0
            raise StopIteration

        idxs = self.indices[self.index : self.index + self.batch_size]
        samples = [self.dataset[i] for i in idxs]
        self.index += self.batch_size

        x_batch, y_batch = zip(*samples)

        return [str(i) for i in idxs], list(x_batch), list(y_batch), list(y_batch)


def create_dataset(cfg: CfgNode):
    def load_with_frame_labels(dataset: PoseTASDataset):
        processed_samples = []
        for i in range(len(dataset)):
            x, y = dataset[i]
            # Expand scalar label into per-frame label if needed
            if y.ndim == 0 or y.shape == ():  # scalar label
                y = torch.full((x.shape[0],), int(y.item()), dtype=torch.long)
            processed_samples.append((x, y))
        return processed_samples

    train_dataset = PoseTASDataset(
        list_path=cfg.data.train_list,
        feature_dir=cfg.data.feature_dir,
        label_dir=cfg.data.label_dir,
        max_frames=cfg.train.max_frames
    )

    val_dataset = PoseTASDataset(
        list_path=cfg.data.val_list,
        feature_dir=cfg.data.feature_dir,
        label_dir=cfg.data.label_dir,
        max_frames=cfg.train.max_frames
    )

    test_dataset = None
    if hasattr(cfg.data, "test_list") and os.path.exists(cfg.data.test_list):
        test_dataset = PoseTASDataset(
            list_path=cfg.data.test_list,
            feature_dir=cfg.data.feature_dir,
            label_dir=cfg.data.label_dir,
            max_frames=cfg.train.max_frames
        )

    # Apply label expansion (if needed) — optional, use if your labels are scalar
    # train_dataset = load_with_frame_labels(train_dataset)
    # val_dataset = load_with_frame_labels(val_dataset)
    # test_dataset = load_with_frame_labels(test_dataset) if test_dataset else None

    return train_dataset, val_dataset, test_dataset
