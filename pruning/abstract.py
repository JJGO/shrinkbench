from collections import OrderedDict

from .utils import (get_params,
                    get_activations,
                    get_param_gradients)

from .mask import mask_module, apply_masks


class Pruning:

    def __init__(self, **pruning_params):
        # Register for __repr__
        self.pruning_params = pruning_params
        # Save them for convenience
        for k, v in pruning_params.items():
            setattr(self, k, v)

    def model_masks(self, model, inputs, outputs):
        raise NotImplementedError
        # return masks

    def apply(self, model, inputs=None, outputs=None):
        masks = self.model_masks(model, inputs, outputs)
        return mask_module(model, masks)

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        for k, v in self.pruning_params.items():
            s += f"{k}={v}, "
        s = s[:-2] + ')'
        return s

    def __str__(self):
        return repr(self)


class LayerPruning(Pruning):

    def module_masks(self, module):
        raise NotImplementedError
        # return masks

    def model_masks(self, model, prunable=None):
        # self.params = get_params(model)

        # if inputs is not None:
        #     self.activations = get_activations(model, inputs)

        #     if outputs is not None:
        #         # TODO fix to return also input, output gradients
        #         self.gradients = get_param_gradients(model, inputs, outputs)

        """Compute masks using the said strategy for every module
        This is a straight forward implementation that supports
        strategies that prune each module independently
        """
        masks = OrderedDict()
        modules = prunable if prunable is not None else model.modules()

        for module in modules:
            masks_ = self.module_masks(module)
            if len(masks_) > 0:
                masks[module] = masks_

        return masks


class CompoundPruning(Pruning):

    def __init__(self, pruning_strategies):
        self.pruning_strategies = pruning_strategies

    def model_masks(self, model, inputs=None, outputs=None):
        for strat in self.pruning_strategies:
            masks = strat.model_masks(model, inputs, outputs)
            apply_masks(model, masks)  # Operation is inplace
        return masks

    def __repr__(self):
        s = f"{self.__class__.__name__}([\n"
        for strat in self.pruning_strategies:
            s += f"\t{repr(strat)}, \n"
        s += '])'
        return s
