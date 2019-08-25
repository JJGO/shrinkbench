import random
import numpy as np
import torch


def fix_seed(seed=42):
    # https://pytorch.org/docs/stable/notes/randomness.html

    # Python
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
