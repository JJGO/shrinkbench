""" Module with examples of common pruning patterns
"""
from .abstract import Pruning
from .utils import get_activations, get_param_gradients


class ActivationMixin(Pruning):

    def update_activations(self):
        assert self.inputs is not None, \
            "Inputs must be provided for activations"
        self._activations = get_activations(self.model, self.inputs)

    def activations(self, only_prunable=True):
        if not hasattr(self, '_activations'):
            self.update_activations()
        if only_prunable:
            return {module: self._activations[module] for module in self.prunable}
        else:
            return self._activations

    def module_activations(self, module):
        if not hasattr(self, '_activations'):
            self.update_activations()
        return self._activations[module]


class GradientMixin(Pruning):

    def update_gradients(self):
        assert self.inputs is not None and self.outputs is not None, \
            "Inputs and Outputs must be provided for gradients"
        self._param_gradients = get_param_gradients(self.model, self.inputs, self.outputs)

    def param_gradients(self, only_prunable=True):
        if not hasattr(self, "_param_gradients"):
            self.update_gradients()
        if only_prunable:
            return {module: self._param_gradients[module] for module in self.prunable}
        else:
            return self._param_gradients

    def module_param_gradients(self, module):
        if not hasattr(self, "_param_gradients"):
            self.update_gradients()
        return self._param_gradients[module]

    def input_gradients(self):
        raise NotImplementedError("Support coming soon")

    def output_gradients(self):
        raise NotImplementedError("Support coming soon")
