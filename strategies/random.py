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
        super(RandomPruning, self).__init__(compression=compression,
                                            prune_classifier=prune_classifier)
        self.masked_modules = (nn.Linear, nn.Conv2d)

    def model_masks(self, model, *_):

        prunable = prunable_modules(model,
                                    self.masked_modules,
                                    self.prune_classifier)
        fraction = fraction_to_keep(self.compression, model, prunable)

        masks = { name : { k : random_mask(v, fraction)
                           for k, v in get_params(m).items()}
                  for name, m in prunable.items() }

        # if not self.prune_classifier:
        #     masks = unmask_classifier(model, masks)
        return masks

# class RandomPruning:

#     masked_modules = (nn.Linear, nn.Conv2d)

#     def __init__(self, fraction):
#         self.fraction = fraction

#     def _idx(self, tensor):
#         idx = np.random.uniform(0, 1, size=tensor.shape) > self.fraction
#         return idx

#     def mask(self, tensor):
#         slice_ = self._idx(tensor)
#         mask = np.ones_like(tensor)
#         mask[slice_] = 0.0
#         return mask

#     def prune(self, tensor, inplace=True):
#         pruned_tensor = tensor if inplace else tensor.copy()
#         slice_ = self._idx(tensor)
#         pruned_tensor[slice_] = 0.0

#         return pruned_tensor

#     def module_masks(self, module):
#         masks = {}
#         if isinstance(module, RandomPruning.masked_modules):
#             masks['weight'] = self.mask(module.weight.detach().cpu().numpy())
#             if module.bias is not None:
#                 masks['bias'] = self.mask(module.bias.detach().cpu().numpy())
#         return masks

#     def __repr__(self):
#         return f"{self.__class__.__name__}(fraction={self.fraction}, axis={self.axis})"

#     def __str__(self):
#         return repr(self)

#     def shortrepr(self):
#         return f"rnd_f{str(self.fraction)[2:]}"
