import numpy as np
import torch


def nonzero(tensor):
    """Returns absolute number of values different from 0

    Parameters
    ----------
    tensor : ndarray

    Returns
    -------
    int
        Number of nonzero elements
    """
    return np.sum(tensor != 0.0)

# https://pytorch.org/docs/stable/tensor_attributes.html
dtype2bits = {
    torch.float32: 32,
    torch.float: 32,
    torch.float64: 64,
    torch.double: 64,
    torch.float16: 16,
    torch.half: 16,
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.short: 16,
    torch.int32: 32,
    torch.int: 32,
    torch.int64: 64,
    torch.long: 64,
    torch.bool: 1,
}


from .size import model_size, model_size_bits
from .accuracy import accuracy, correct
from .memory import memory_size