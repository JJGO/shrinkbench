import numpy as np


def abs_threshold(tensor, fraction):
    assert isinstance(tensor, np.ndarray)
    # fraction to keep
    size = np.prod(tensor.shape)
    raveled_val = np.sort(-np.abs(tensor), axis=None)
    threshold = np.abs(raveled_val[int(size*fraction)])
    return threshold


def largest_abs_mask(tensor, fraction=None, threshold=None):
    assert isinstance(tensor, np.ndarray)
    # fraction to keep
    assert (fraction is None) ^ (threshold is None), \
        "Either fraction or threshold must be provided"

    if threshold is None:
        threshold = abs_threshold(tensor, fraction)
    idx = np.logical_and(tensor < threshold, tensor > -threshold)
    mask = np.ones_like(tensor)
    mask[idx] = 0
    return mask


def norms_tensor(tensor, ord, matrix_mode=False):
    if matrix_mode:
        assert len(tensor.shape) > 2
        tensor_flattened = tensor.reshape(*tensor.shape[:2], -1)
    else:
        tensor_flattened = tensor.reshape(tensor.shape[0], -1)
    norms = []
    for w in tensor_flattened:
        norms.append(np.linalg.norm(w, ord))
    return np.array(norms)


def largest_norm_channel_mask(tensor, fraction, ord=1, matrix_mode=False):
    # Assume channels is first axis, true for conv & linear
    n_channels = tensor.shape[0]
    norms = norms_tensor(tensor, ord, matrix_mode)
    idx = np.argsort(-norms)
    pruned_channels = idx[int(n_channels*fraction):]
    mask = np.ones_like(tensor)
    mask[pruned_channels, ...] = 0.0
    return mask


def largest_norm_activation_mask(tensor, activation, fraction, ord=1, matrix_mode=False):
    # Sum over batch dimnesion
    activation = activation.sum(axis=0)
    n_channels = tensor.shape[0]
    norms = norms_tensor(activation, ord, matrix_mode)
    idx = np.argsort(-norms)
    prunned_channels = idx[int(n_channels*fraction):]
    mask = np.ones_like(tensor)
    mask[prunned_channels, ...] = 0.0
    return mask
