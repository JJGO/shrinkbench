# -*- coding: utf-8 -*-

"""Magnitude based pruning

Implements pruning strategy that prunes as follows

For a given tensor, compute a
"""

import numpy as np
import torch.nn as nn


class ChannelPruning:

    masked_modules = (nn.Linear, nn.Conv2d)

    def __init__(self, fraction, axis=0):
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
        if slice_ is not None:
            mask[slice_] = 0.0
        return mask

    def prune(self, tensor, inplace=True):
        pruned_tensor = tensor if inplace else tensor.copy()
        slice_ = self._idx(tensor)
        pruned_tensor[slice_] = 0.0

        return pruned_tensor

    def module_masks(self, module, axis=0):
        masks = {}
        if isinstance(module, ChannelPruning.masked_modules):
            masks['weight'] = self.mask(module.weight.detach().cpu().numpy())
            if module.bias is not None:
                # For channel pruning we need to prune the bias
                # corresponding to the pruned channels
                axes = tuple(i for i in range(len(masks['weight'].shape))
                             if i != axis)
                where = masks['weight'].sum(axis=axes) == 0
                bias_mask = np.ones_like(module.bias.detach().cpu().numpy())
                bias_mask[where] = 0
                masks['bias'] = bias_mask

        return masks

    def __repr__(self):
        return f"{self.__class__.__name__}(fraction={self.fraction}, axis={self.axis})"

    def __str__(self):
        return repr(self)

    def shortrepr(self):
        return f"chn_f{str(self.fraction)[2:]}"
