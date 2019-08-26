# -*- coding: utf-8 -*-

"""Random  pruning

Implements pruning strategy that randomly prunes weights.
It is intended as a baseline
"""

import numpy as np
import torch.nn as nn


class RandomPruning:

    masked_modules = (nn.Linear, nn.Conv2d)

    def __init__(self, fraction):
        self.fraction = fraction

    def _idx(self, tensor):
        idx = np.random.uniform(0, 1, size=tensor.shape) > self.fraction
        return idx

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

    def module_masks(self, module):
        masks = {}
        if isinstance(module, RandomPruning.masked_modules):
            masks['weight'] = self.mask(module.weight.detach().cpu().numpy())
            if module.bias is not None:
                masks['bias'] = self.mask(module.bias.detach().cpu().numpy())
        return masks

    def __repr__(self):
        return f"{self.__class__.__name__}(fraction={self.fraction}, axis={self.axis})"

    def __str__(self):
        return repr(self)

    def shortrepr(self):
        return f"rnd_f{str(self.fraction)[2:]}"
