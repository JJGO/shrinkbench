"""Masked versions of common torch.nn.Modules

Implementations of most common parametric torch layers.
For vision classification networks the inmense majority of the
parameters are in either Conv2d layers and Dense Layers (called
Linear in PyTorch)

Variables:
    masked_modules {dict} -- [description]
"""
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


def _same_shape(x_mask, x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    return x.shape == x_mask.shape


class MaskedModule(nn.Module):

    def __init__(self, layer, weight_mask, bias_mask=None):
        super(MaskedModule, self).__init__()

        self.weight = layer.weight
        self.bias = layer.bias

        # use register_buffer so model.to(device) works on fixed tensors like masks
        self.register_buffer("weight_mask", None)
        self.register_buffer("bias_mask", None)

        self.set_masks(weight_mask, bias_mask)

    def forward_pre(self):
        # Masks are pre multiplied, effectively
        # zeroing gradients to masked weights
        weight = self.weight * self.weight_mask
        if self.bias_mask is not None:
            bias = self.bias * self.bias_mask
        else:
            bias = self.bias
        return weight, bias

    def set_masks(self, weight_mask, bias_mask=None):
        assert _same_shape(weight_mask, self.weight), f"Weight Mask must match dimensions"

        # Multiply weights by masks so metrics can count nonzeros
        weight_mask = _ensure_tensor(weight_mask)
        self.weight_mask = _same_device(weight_mask, self.weight)
        self.weight.data.mul_(weight_mask)

        if bias_mask is not None:
            bias_mask = _ensure_tensor(bias_mask)
            assert self.bias is not None, "Provided layer must have bias for it to be masked"
            assert _same_shape(bias_mask, self.bias), f"Bias Mask must match dimensions"
            self.bias_mask = _same_device(bias_mask, self.bias)
            self.bias.data.mul_(bias_mask)


class LinearMasked(MaskedModule):

    def __init__(self, linear_layer, weight_mask, bias_mask=None):
        """Masked version of a linear layer for pruning evaluation

        Constructed from an existing layer, a weight mask (and optionally
        a bias mask). By construction ensures backpropagation does not change
        masked parameters so they stay at zero.

        Arguments:
            linear_layer {torch.nn.Linear} -- Layer to mask. Not modified.
            weight_mask {numpy.ndarray} -- Mask with zero entries for weight vector

        Keyword Arguments:
            bias_mask {numpy.ndarray} -- Mask with zero entries for bias vector (default: {None})
        """
        super(LinearMasked, self).__init__(linear_layer, weight_mask, bias_mask)
        assert isinstance(linear_layer, nn.Linear), "Layer must be a linear layer"
        for attr in ['in_features', 'out_features']:
            setattr(self, attr, getattr(linear_layer, attr))

    def forward(self, input):
        weight, bias = self.forward_pre()
        return F.linear(input, weight, bias)

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        s += f'in_features={self.in_features}, '
        s += f'out_features={self.out_features}, '
        s += f'bias={self.bias is not None})'
        return s


class Conv2dMasked(MaskedModule):

    def __init__(self, conv_layer, weight_mask, bias_mask=None):
        """Masked version  of 2D convolutional layer for pruning evaluation

        Constructed from an existing layer, a weight mask (and optionally
        a bias mask). By construction ensures backpropagation does not change
        masked parameters so they stay at zero.

        [description]

        Arguments:
            linear_layer {torch.nn.Conv2d} -- Layer to mask. Not modified.
            weight_mask {numpy.ndarray} -- Mask with zero entries for weight vector

        Keyword Arguments:
            bias_mask {numpy.ndarray} -- Mask with zero entries for bias vector (default: {None})
        """
        super(Conv2dMasked, self).__init__(conv_layer, weight_mask, bias_mask)
        assert isinstance(conv_layer, nn.Conv2d), "Layer must be a Conv2d layer"
        for attr in ['in_channels', 'out_channels', 'kernel_size', 'dilation',
                     'stride', 'padding', 'padding_mode', 'groups']:
            setattr(self, attr, getattr(conv_layer, attr))

    def forward(self, input):
        weight, bias = self.forward_pre()
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        s += ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
              ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(**self.__dict__)

# TODO Conv1D Conv3D ConvTranspose
# squeeze out Convs for channel pruning


masked_modules = {
    nn.Linear: LinearMasked,
    nn.Conv2d: Conv2dMasked,
}
