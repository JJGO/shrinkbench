import numpy as np

from . import nonzero, dtype2bits
from ..pruning.utils import get_activations


def memory_size(model, input, as_bits=False):
    """Compute memory size estimate

    Note that this is computed for training purposes, since
    all input activations to parametric are accounted for.

    For inference time you can free memory as you go but with
    residual connections you are forced to remember some, thus
    this is left to implement (TODO)

    The input is required in order to materialize activations
    for dimension independent layers. E.g. Conv layers work
    for any height width.

    Arguments:
        model {torch.nn.Module} -- [description]
        input {torch.Tensor} --

    Keyword Arguments:
        as_bits {bool} -- [description] (default: {False})

    Returns:
        tuple:
         - int -- Estimated memory needed for the full model
         - int -- Estimated memory needed for nonzero activations
    """
    batch_size = input.size(0)
    total_memory = nonzero_memory = np.prod(input.shape)

    activations = get_activations(model, input)

    # TODO only count parametric layers
    # Input activations are the ones we need for backprop
    input_activations = [i for _, (i, o) in activations.items()]
    for act in input_activations:
        t = np.prod(act.shape)
        nz = nonzero(act)
        if as_bits:
            bits = dtype2bits[act.dtype]
            t *= bits
            nz *= bits
        total_memory += np.prod(act.shape)
        nonzero_memory += nonzero(act)

    return total_memory/batch_size, nonzero_memory/batch_size
