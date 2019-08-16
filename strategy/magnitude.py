# -*- coding: utf-8 -*-

"""Magnitude based pruning

Implements pruning strategy that prunes as follows

For a given tensor, sort by absolute value and
keep only the _fraction_ with highest magnitudes.
"""

import numpy as np


class MagnitudePruning:

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

    def match(self, param):
        if param.endswith('.weight'):
            return True
        elif param.endswith('.bias'):
            return True
        return False
