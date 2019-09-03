# -*- coding: utf-8 -*-

"""Random  pruning

Implements pruning strategy that randomly prunes weights.
It is intended as a baseline
"""

import numpy as np
import torch.nn as nn


from ..pruning import *


def random_mask(tensor, fraction):
    idx = np.random.uniform(0, 1, size=tensor.shape) > fraction
    mask = np.ones_like(tensor)
    mask[idx] = 0.0
    return mask


class RandomPruning(Pruning):

    def __init__(self, compression, prune_classifier=False):
        super(RandomPruning, self).__init__(
            compression=compression,
            prune_classifier=prune_classifier)
        self.masked_modules = (nn.Linear, nn.Conv2d)

    def model_masks(self, model, *_):

        prunable = prunable_modules(model,
                                    self.masked_modules,
                                    self.prune_classifier)
        fraction = fraction_to_keep(self.compression, model, prunable)

        masks = { module : { k : random_mask(v, fraction)
                           for k, v in get_params(module).items()}
                  for module in prunable }

        return masks
