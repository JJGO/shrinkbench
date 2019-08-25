# -*- coding: utf-8 -*-

"""Model size metrics
"""

import numpy as np

from . import nonzero, dtype2bits


def model_size(model):

    total_params = 0
    nonzero_params = 0
    for param, tensor in model.state_dict().items():
        if True:
            total_params += np.prod(tensor.shape)
            nonzero_params += nonzero(tensor.cpu().numpy())
    return int(total_params), int(nonzero_params)


def model_size_bits(model):

    total_params = 0
    nonzero_params = 0
    for param, tensor in model.state_dict().items():
        total_params += np.prod(tensor.shape) * dtype2bits[tensor.dtype]
        nonzero_params += nonzero(tensor.cpu().numpy()) * dtype2bits[tensor.dtype]
    return int(total_params), int(nonzero_params)


# param.endswith('weight') or param.endswith('bias'):

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
