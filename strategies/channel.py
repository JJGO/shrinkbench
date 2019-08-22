# -*- coding: utf-8 -*-

"""Magnitude based pruning

Implements pruning strategy that prunes as follows

For a given tensor, compute a
"""

import numpy as np


class ChannelPruning:

    def __init__(self, fraction, axis=1):
        self.fraction = fraction
        self.axis = axis

    def _idx(self, tensor):
        axis = self.axis
        n_dims = len(tensor.shape)
        n_channels = tensor.shape[axis]

        # Sum in all but channel axis
        axes = tuple(i for i in range(n_dims) if i != axis)
        channel_sums = np.sum(np.abs(tensor), axis=axes)

        # Sort by descending abs activation
        channel_idx = np.argsort(-channel_sums)

        # Zero out 1-fraction by slicing across just the axis dim and setting to 0
        pruned_channels = channel_idx[int(n_channels*self.fraction):]
        slice_ = [pruned_channels if i == axis else slice(None)
                  for i in range(n_dims)]
        return slice_

    def mask(self, tensor):
        slice_ = self._idx(tensor)
        mask = np.ones_like(tensor)
        mask[slice_] = 0.0
        return mask

    def prune(self, tensor, inplace=True):
        pruned_tensor = tensor if inplace else tensor.copy()
        slice_ = self._idx(tensor)
        pruned_tensor[slice_] = 0.0

        return pruned_tensor

    def match(self, param):
        if param.endswith('.weight'):
            return True
        return False

    def __repr__(self):
        return f"{self.__class__.__name__}(fraction={self.fraction}, axis={self.axis})"

    def __str__(self):
        return repr(self)

    def shortrepr(self):
        return f"chn_f{int(100*self.fraction)}"
