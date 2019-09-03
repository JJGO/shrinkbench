# -*- coding: utf-8 -*-

"""Magnitude based pruning

Implements pruning strategy that prunes as follows

For a given tensor, sort by absolute value and
keep only the _fraction_ with highest magnitudes.
"""

import numpy as np
import torch.nn as nn

from ..pruning import *


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


class GlobalMagnitudePruning(Pruning):

    def __init__(self, compression, prune_classifier=False):
        super(GlobalMagnitudePruning, self).__init__(compression=compression,
                                                     prune_classifier=prune_classifier)
        self.masked_modules = (nn.Linear, nn.Conv2d)

    def model_masks(self, model, *_):

        prunable = prunable_modules(model,
                                    self.masked_modules,
                                    self.prune_classifier)
        fraction = fraction_to_keep(self.compression, model, prunable)

        params = {k: v for name, module in prunable.items()
                  for k, v in get_params(module, prefix=name).items()}

        flat_params = np.concatenate([v.flatten() for v in params.values()])
        threshold = abs_threshold(flat_params, fraction)

        masks = { name : { k : largest_abs_mask(v, threshold=threshold)
                           for k, v in get_params(m).items()}
                  for name, m in prunable.items() }

        return masks


class LayerMagnitudePruning(LayerPruning):

    def __init__(self, compression, prune_classifier=False):
        super(LayerMagnitudePruning, self).__init__(compression=compression,
                                                    prune_classifier=prune_classifier)
        self.masked_modules = (nn.Linear, nn.Conv2d)

    def module_masks(self, module):
        masks = {}
        if isinstance(module, self.masked_modules):
            params = get_params(module)
            masks['weight'] = largest_abs_mask(params['weight'], fraction=self.fraction)
            if module.bias is not None:
                masks['bias'] = largest_abs_mask(params['bias'], fraction=self.fraction)
        return masks

    def model_masks(self, model, *_):
        self.prunable = prunable_modules(model,
                                         self.masked_modules,
                                         self.prune_classifier)
        self.fraction = fraction_to_keep(self.compression, model, self.prunable)
        masks = super(LayerMagnitudePruning, self).model_masks(model, None, None)
        delattr(self, 'fraction')
        delattr(self, 'prunable')
        return masks
