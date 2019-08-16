# -*- coding: utf-8 -*-

"""Model size metrics
"""

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


def model_size(model):
    """Returns absolute number of values different from 0

    Parameters
    ----------
    model : pytorch model

    Returns
    -------
    total_params : int
        Total number of weight & bias params
    nonzero_params : int
        Out total_params exactly how many are nonzero
    """

    # TODO look at dtype to figure out the exact model size

    total_params = 0
    nonzero_params = 0
    for param, tensor in model.state_dict().items():
        if param.endswith('weight') or param.endswith('bias'):
            total_params += np.prod(tensor.shape)
            nonzero_params += nonzero(tensor.cpu().numpy())
    return total_params, nonzero_params
