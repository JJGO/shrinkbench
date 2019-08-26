import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def _ensure_tensor(x):
    # Aux functions in case mask arguments are numpy arrays
    if not isinstance(x, torch.Tensor) and x is not None:
        return torch.from_numpy(x)
    return x


def _same_device(x_mask, x):
    # Aux function to ensure same device fo weight and mask
    # so _mul doesn't fail
    if x.device != x_mask.device:
        return x_mask.to(x.device)
    return x_mask


class LinearMasked(nn.Module):

    def __init__(self, linear_layer, weight_mask, bias_mask=None):
        super(LinearMasked, self).__init__()

        self.linear_layer = linear_layer
        self.weight = self.linear_layer.weight
        self.bias = self.linear_layer.bias

        # use register_buffer so model.to(device) works on fixed tensors like masks
        self.register_buffer("weight_mask", _ensure_tensor(weight_mask))
        self.register_buffer("bias_mask", _ensure_tensor(bias_mask))

        # Multiply weights by masks so metrics can count nonzeros
        self.weight_mask = _same_device(self.weight_mask, self.weight)
        self.weight.data.mul_(self.weight_mask)

        if bias_mask is not None:
            self.bias_mask = _same_device(self.bias_mask, self.bias)
            self.bias.data.mul_(self.bias_mask)
            assert linear_layer.bias is not None

    def forward(self, input):
        weight = self.weight * self.weight_mask
        if self.bias_mask is not None:
            bias = self.bias * self.bias_mask
        else:
            bias = self.bias
        return F.linear(input, weight, bias)


class Conv2dMasked(nn.Module):

    def __init__(self, conv_layer, weight_mask, bias_mask=None):
        super(Conv2dMasked, self).__init__()

        self.conv_layer = conv_layer
        self.weight = self.conv_layer.weight
        self.bias = self.conv_layer.bias

        # use register_buffer so model.to(device) works on fixed tensors like masks
        self.register_buffer("weight_mask", _ensure_tensor(weight_mask))
        self.register_buffer("bias_mask", _ensure_tensor(bias_mask))

        # Multiply weights by masks so metrics can count nonzeros
        self.weight_mask = _same_device(self.weight_mask, self.weight)
        self.weight.data.mul_(self.weight_mask)

        if bias_mask is not None:
            self.bias_mask = _same_device(self.bias_mask, self.bias)
            self.bias.data.mul_(self.bias_mask)
            assert conv_layer.bias is not None

    def forward(self, input):
        conv = self.conv_layer
        weight = self.weight * self.weight_mask
        if self.bias_mask is not None:
            bias = self.bias * self.bias_mask
        else:
            bias = self.bias

        if conv.padding_mode == 'circular':
            expanded_padding = ((conv.padding[1] + 1) // 2, conv.padding[1] // 2,
                                (conv.padding[0] + 1) // 2, conv.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, bias, conv.stride,
                            _pair(0), conv.dilation, conv.groups)
        return F.conv2d(input, weight, bias, conv.stride,
                        conv.padding, conv.dilation, conv.groups)


masked_modules = {
    nn.Linear: LinearMasked,
    nn.Conv2d: Conv2dMasked,
}
