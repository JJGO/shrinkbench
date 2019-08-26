# -*- coding: utf-8 -*-

"""Magnitude based pruning

Implements pruning strategy that prunes as follows

For a given tensor, sort by absolute value and
keep only the _fraction_ with highest magnitudes.
"""

import numpy as np
import torch.nn as nn


class MagnitudePruning:

    masked_modules = (nn.Linear, nn.Conv2d)

    def __init__(self, fraction):
        self.fraction = fraction

    def _idx(self, tensor):
        size = np.prod(tensor.shape)
        # Get magnitude and sort based on it
        raveled_idx = np.argsort(-np.abs(tensor), axis=None)
        # Prune the 1-fraction, unravel & set to 0
        pruned_idx = raveled_idx[int(size*self.fraction):]
        unraveled_idx = np.unravel_index(pruned_idx, tensor.shape)
        return unraveled_idx

    def mask(self, tensor):
        unraveled_idx = self._idx(tensor)
        mask = np.ones_like(tensor)
        mask[unraveled_idx] = 0
        return mask

    def prune(self, tensor, inplace=True):
        pruned_tensor = tensor if inplace else tensor.copy()
        unraveled_idx = self._idx(tensor)
        pruned_tensor[unraveled_idx] = 0

        return pruned_tensor

    def module_masks(self, module):
        masks = {}
        if isinstance(module, MagnitudePruning.masked_modules):
            masks['weight'] = self.mask(module.weight.detach().cpu().numpy())
            if module.bias is not None:
                masks['bias'] = self.mask(module.bias.detach().cpu().numpy())
        return masks

    def __repr__(self):
        return f"{self.__class__.__name__}(fraction={self.fraction})"

    def __str__(self):
        return repr(self)

    def shortrepr(self):
        return f"mag_f{str(self.fraction)[2:]}"
