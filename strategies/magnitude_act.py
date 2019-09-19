# -*- coding: utf-8 -*-


import numpy as np
import torch.nn as nn

from ..pruning import *
from .utils import abs_threshold, largest_abs_mask, norms_tensor


def importance(weight, activation, norm=1):
    # TODO do properly
    activation = activation.mean(axis=0)  # reduce batch dim
    norms = norms_tensor(activation, norm)
    # add filter dims
    if len(weight.shape) > 2:
        # normalize spatial dims
        norms /= activation.shape[1] * activation.shape[2]
        # TODO refactor with np.tile or sth
        norms = norms[..., np.newaxis, np.newaxis]
        norms = np.repeat(norms, weight.shape[2], axis=1)
        norms = np.repeat(norms, weight.shape[3], axis=2)
        # normalize across spatial dimensions


    if len(weight.shape) > len(norms.shape):
        # broadcasting will take care of it
        return weight * norms
    else:
        # for bias
        return weight * norms.mean()


class GlobalActivationMagnitudePruning(Pruning):

    def __init__(self, compression, prune_classifier=False):
        super(GlobalActivationMagnitudePruning, self).__init__(compression=compression,
                                                               prune_classifier=prune_classifier)
        self.masked_modules = (nn.Linear, nn.Conv2d)

    def model_masks(self, model, inputs, outputs):

        prunable = prunable_modules(model,
                                    self.masked_modules,
                                    self.prune_classifier)
        fraction = fraction_to_keep(self.compression, model, prunable)

        activations = get_activations(model, inputs)

        flat_params_times_grads = np.concatenate([
            importance(v, activations[module][0]).flatten()
            for module in prunable
                for k, v in get_params(module).items()
            ])

        threshold = abs_threshold(flat_params_times_grads, fraction)

        masks = { mod :{ k : largest_abs_mask(importance(v, activations[mod][0]), threshold=threshold)
                    for k, v in get_params(mod).items() }
                  for mod in prunable}
        return masks


class LayerActivationMagnitudePruning(LayerPruning):

    def __init__(self, compression, prune_classifier=False):
        super(LayerActivationMagnitudePruning, self).__init__(compression=compression,
                                                              prune_classifier=prune_classifier)
        self.masked_modules = (nn.Linear, nn.Conv2d)

    def module_masks(self, module):
        masks = {}
        activations = self.activations[module][0]
        if isinstance(module, self.masked_modules):
            params = get_params(module)
            masks['weight'] = largest_abs_mask(importance(params['weight'], activations), fraction=self.fraction)
            if module.bias is not None:
                masks['bias'] = largest_abs_mask(importance(params['bias'], activations), fraction=self.fraction)
        return masks

    def model_masks(self, model, inputs, outputs):
        prunable = prunable_modules(model,
                                    self.masked_modules,
                                    self.prune_classifier)
        self.fraction = fraction_to_keep(self.compression, model, prunable)
        self.activations = get_activations(model, inputs)
        masks = super(LayerActivationMagnitudePruning, self).model_masks(model, prunable=prunable)
        delattr(self, 'fraction')
        delattr(self, 'activations')
        return masks
