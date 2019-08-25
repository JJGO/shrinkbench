# -*- coding: utf-8 -*-

"""Random  pruning

Implements pruning strategy that randomly prunes weights.
It is intended as a baseline
"""

import numpy as np


class RandomPruning:

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

    def match(self, param):
        return False

    def __repr__(self):
        return f"{self.__class__.__name__}(fraction={self.fraction}, axis={self.axis})"

    def __str__(self):
        return repr(self)

    def shortrepr(self):
        return f"rnd_f{int(100*self.fraction)}"
