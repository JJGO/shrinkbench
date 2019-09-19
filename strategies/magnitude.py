# -*- coding: utf-8 -*-

"""Magnitude based pruning

Implements pruning strategy that prunes as follows

For a given tensor, sort by absolute value and
keep only the _fraction_ with highest magnitudes.
"""

import numpy as np
import torch.nn as nn

from ..pruning import *
from .utils import abs_threshold, largest_abs_mask


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

        flat_params = np.concatenate([
            param.flatten() for module in prunable
            for _, param in get_params(module).items()
            ])

        threshold = abs_threshold(flat_params, fraction)

        masks = { module :
                  { k : largest_abs_mask(v, threshold=threshold)
                    for k, v in get_params(module).items() }
                  for module in prunable}
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
        prunable = prunable_modules(model,
                                    self.masked_modules,
                                    self.prune_classifier)
        self.fraction = fraction_to_keep(self.compression, model, prunable)
        masks = super(LayerMagnitudePruning, self).model_masks(model, prunable=prunable)
        delattr(self, 'fraction')
        return masks
