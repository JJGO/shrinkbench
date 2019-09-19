import numpy as np

from . import nonzero, dtype2bits
from ..pruning.utils import get_activations


def memory_size(model, input):
    # TODO think about memory size in more proper terms (inference vs training)
    batch_size = input.size(0)
    total_memory = nonzero_memory = np.prod(input.shape)

    activations = get_activations(model, input)

    # The ones we need for backprop
    input_activations = [i for _, (i, o) in activations.items()]
    for act in input_activations:
        total_memory += np.prod(act.shape)
        nonzero_memory += nonzero(act)

    return total_memory/batch_size, nonzero_memory/batch_size

# TODO add bits flag version