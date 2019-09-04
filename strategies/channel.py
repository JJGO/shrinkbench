# -*- coding: utf-8 -*-

"""Magnitude based pruning

Implements pruning strategy that prunes as follows

For a given tensor, compute a
"""

import numpy as np
import torch.nn as nn

from ..pruning import *
from .utils import largest_norm_channel_mask, largest_norm_activation_mask


# TODO Channel pruning does not propagage zeroing out stuff
# TODO These are just output act based methods, input act based methods are
# to be implemented

class WeightNormChannelPruning(LayerPruning):

    def __init__(self, compression, norm=1, prune_linear=True):
        super(WeightNormChannelPruning, self).__init__(
            compression=compression,
            norm=norm,
            prune_linear=prune_linear)

        self.masked_modules = (nn.Conv2d,)
        if self.prune_linear:
            self.masked_modules += (nn.Linear,)

    def module_masks(self, module):
        masks = {}
        if isinstance(module, self.masked_modules):
            params = get_params(module)
            masks['weight'] = largest_norm_channel_mask(params['weight'],
                                                        self.fraction,
                                                        self.norm)
            if module.bias is not None:
                # Mask associated bias
                axes = tuple(range(len(params['weight'].shape)))[1:]
                masks['bias'] = masks['weight'].mean(axis=(axes))
        return masks

    def model_masks(self, model, *_):
        self.prunable = prunable_modules(model,
                                         self.masked_modules,
                                         False)
        self.fraction = fraction_to_keep(self.compression, model, self.prunable)
        masks = super(WeightNormChannelPruning, self).model_masks(model, None, None)
        delattr(self, 'fraction')
        delattr(self, 'prunable')
        return masks


class ActivationNormChannelPruning(LayerPruning):

    def __init__(self, compression, norm=1, prune_linear=True):
        super(ActivationNormChannelPruning, self).__init__(
            compression=compression,
            norm=norm,
            prune_linear=prune_linear)

        self.masked_modules = (nn.Conv2d,)
        if self.prune_linear:
            self.masked_modules += (nn.Linear,)

    def module_masks(self, module):
        masks = {}
        if isinstance(module, self.masked_modules):
            params = get_params(module)
            _, out_activation = self.activations[module]

            masks['weight'] = largest_norm_activation_mask(params['weight'],
                                                           out_activation,
                                                           self.fraction,
                                                           self.norm)
            if module.bias is not None:
                # Mask associated bias
                axes = tuple(range(len(params['weight'].shape)))[1:]
                masks['bias'] = masks['weight'].mean(axis=(axes))
        return masks

    def model_masks(self, model, inputs, *_):
        self.prunable = prunable_modules(model,
                                         self.masked_modules,
                                         False)
        self.fraction = fraction_to_keep(self.compression, model, self.prunable)
        masks = super(ActivationNormChannelPruning, self).model_masks(model, inputs, None)
        delattr(self, 'fraction')
        delattr(self, 'prunable')
        return masks


