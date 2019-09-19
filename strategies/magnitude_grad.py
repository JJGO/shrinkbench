# -*- coding: utf-8 -*-


import numpy as np
import torch.nn as nn

from ..pruning import *
from .utils import abs_threshold, largest_abs_mask


class GlobalGradMagnitudePruning(Pruning):

    def __init__(self, compression, prune_classifier=False):
        super(GlobalGradMagnitudePruning, self).__init__(compression=compression,
                                                         prune_classifier=prune_classifier)
        self.masked_modules = (nn.Linear, nn.Conv2d)

    def model_masks(self, model, inputs, outputs):

        prunable = prunable_modules(model,
                                    self.masked_modules,
                                    self.prune_classifier)
        fraction = fraction_to_keep(self.compression, model, prunable)

        gradients = get_param_gradients(model, inputs, outputs)

        flat_params_times_grads = np.concatenate([
            (v * gradients[module][k]).flatten()
            for module in prunable
                for k, v in get_params(module).items()
            ])

        threshold = abs_threshold(flat_params_times_grads, fraction)

        masks = { mod :{ k : largest_abs_mask(v * gradients[mod][k], threshold=threshold)
                    for k, v in get_params(mod).items() }
                  for mod in prunable}
        return masks


class LayerGradMagnitudePruning(LayerPruning):

    def __init__(self, compression, prune_classifier=False):
        super(LayerGradMagnitudePruning, self).__init__(compression=compression,
                                                        prune_classifier=prune_classifier)
        self.masked_modules = (nn.Linear, nn.Conv2d)

    def module_masks(self, module):
        masks = {}
        gradients = self.gradients[module]
        if isinstance(module, self.masked_modules):
            params = get_params(module)
            masks['weight'] = largest_abs_mask(params['weight']*gradients['weight'], fraction=self.fraction)
            if module.bias is not None:
                masks['bias'] = largest_abs_mask(params['bias']*gradients['bias'], fraction=self.fraction)
        return masks

    def model_masks(self, model, inputs, outputs):
        prunable = prunable_modules(model,
                                    self.masked_modules,
                                    self.prune_classifier)
        self.fraction = fraction_to_keep(self.compression, model, prunable)
        self.gradients = get_param_gradients(model, inputs, outputs)
        masks = super(LayerGradMagnitudePruning, self).model_masks(model, prunable=prunable)
        delattr(self, 'fraction')
        delattr(self, 'gradients')
        return masks
