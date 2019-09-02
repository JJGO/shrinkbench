import random
import torch
from torch.utils.data import Dataset


class Zeros224(Dataset):

    def __init__(self, num_samples):
        self.len = num_samples

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x = torch.zeros(3, 224, 224)
        y = 0  # random.randint(0, 999)
        return x, y


def train_dataset(preproc=True, path=None):
    return Zeros224(int(1.28e6))


def val_dataset(preproc=True, path=None):
    return Zeros224(int(5e4))
