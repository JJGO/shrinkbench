import numpy as np
from torch import nn

from . import nonzero, dtype2bits
from .raw_flops import dense_flops, conv2d_flops
from ..pruning.utils import get_activations, get_params
from ..pruning import Conv2dMasked, LinearMasked


def _conv2d_flops(module, activation):
    input_shape = activation.shape[2:]  # Ignore batch & channels
    # TODO fix dilation and padding size
    return conv2d_flops(in_channels=module.in_channels,
                        out_channels=module.out_channels,
                        input_shape=input_shape,
                        kernel_shape=module.kernel_size,
                        padding=module.padding_mode,
                        strides=module.stride,
                        dilation=module.dilation)


def _linear_flops(module, activation):
    return dense_flops(module.in_features, module.out_features)


def flops(model, input):

    FLOP_fn = {
        nn.Conv2d: _conv2d_flops,
        Conv2dMasked: _conv2d_flops,
        nn.Linear: _linear_flops,
        LinearMasked: _linear_flops,
    }

    # Multiply-add flops
    total_flops = nonzero_flops = 0
    activations = get_activations(model, input)

    # The ones we need for backprop
    for m, (act, _) in activations.items():
        if m.__class__ in FLOP_fn:
            w = m.weight.detach().cpu().numpy()
            module_flops = FLOP_fn[m.__class__](m, act)
            total_flops += module_flops
            nonzero_flops += module_flops * nonzero(w).sum() / np.prod(w.shape)

    return total_flops, nonzero_flops
