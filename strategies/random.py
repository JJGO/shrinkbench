"""Random  pruning

Implements pruning strategy that randomly prunes weights.
It is intended as a baseline for pruning evalution
"""

import numpy as np
from ..pruning import VisionPruning
from .utils import map_importances


def random_mask(tensor, fraction):
    idx = np.random.uniform(0, 1, size=tensor.shape) > fraction
    mask = np.ones_like(tensor)
    mask[idx] = 0.0
    return mask


class RandomPruning(VisionPruning):

    def model_masks(self):
        params = self.params()
        masks = map_importances(lambda x: random_mask(x, self.fraction), params)
        return masks
