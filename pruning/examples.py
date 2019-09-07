from .abstract import Pruning
from .utils import (get_params,
                    get_activations,
                    get_gradients,
                    get_param_gradients)


class WeightBasedPruning(Pruning):

    def __init__(self, model, **pruning_params):
        super(WeightBasedPruning, self).__init__(model, **pruning_params)

    def model_masks(self, model, *_):

        params = get_params(model)
        raise NotImplementedError
        # return masks


class ActivationBasedPruning(Pruning):

    def __init__(self, model, **pruning_params):
        super(ActivationBasedPruning, self).__init__(model, **pruning_params)

    def model_masks(self, model, inputs, *_):
        params = get_params(model)
        activations = get_activations(model, inputs)
        raise NotImplementedError
        # return masks


class GradientBasedPruning(Pruning):

    def __init__(self, model, **pruning_params):
        super(GradientBasedPruning, self).__init__(model, **pruning_params)

    def model_masks(self, model, inputs, outputs):
        params = get_params(model)
        activations = get_activations(model, inputs)
        gradients = get_gradients(model, inputs, outputs)
        raise NotImplementedError
        # return masks