import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class LinearMasked(nn.Module):

    def __init__(self, linear_layer, weight_mask, bias_mask=None):
        super(LinearMasked, self).__init__()

        self.linear_layer = linear_layer
        self.weight = self.linear_layer.weight
        self.bias = self.linear_layer.bias

        # use register_buffer so model.to(device) works on fixed tensors like masks
        self.register_buffer("weight_mask", weight_mask)
        self.register_buffer("bias_mask", bias_mask)
        if bias_mask is not None:
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
        self.register_buffer("weight_mask", weight_mask)
        self.register_buffer("bias_mask", bias_mask)
        if bias_mask is not None:
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
