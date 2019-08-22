import numpy as np


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


from .size import model_size
from .accuracy import accuracy