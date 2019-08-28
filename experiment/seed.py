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


    # TODO Enable full deterministic computation
    # https://pytorch.org/docs/stable/notes/randomness.html
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
