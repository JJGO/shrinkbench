import warnings
import torch.nn as nn
from .abstract import Pruning
from .utils import fraction_to_keep


class VisionPruning(Pruning):

    def __init__(self, model, inputs=None, outputs=None, compression=1):
        super(VisionPruning).__init__(model, inputs, outputs, compression=compression)
        self.prunable = self.prunable_modules(self.model)
        self.fraction = fraction_to_keep(self.compression, self.model, self.prunable)

    def can_prune(self, module):

        if hasattr(module, 'is_classifier'):
            return module.is_classifier
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            return True
        return False

    def prunable_modules(self, model):
        if not any(getattr(module, 'is_classifier', False) for module in model.modules):
            warnings.warn('No classifier layer found. Pruning classifier is often \
                not desired since it makes some classes to have always zero output')
        return super(VisionPruning, self).prunable_modules(model)
