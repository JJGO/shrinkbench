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

        self.weight = linear_layer.weight
        self.bias = linear_layer.bias

        # use register_buffer so model.to(device) works on fixed tensors like masks
        self.register_buffer("weight_mask", _ensure_tensor(weight_mask))
        self.register_buffer("bias_mask", _ensure_tensor(bias_mask))

        # Multiply weights by masks so metrics can count nonzeros
        self.weight_mask = _same_device(self.weight_mask, self.weight)
        self.weight.data.mul_(self.weight_mask)

        if bias_mask is not None:
            self.bias_mask = _same_device(self.bias_mask, self.bias)
            self.bias.data.mul_(self.bias_mask)
            assert self.bias is not None

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

    def forward(self, input):
        weight = self.weight * self.weight_mask
        if self.bias_mask is not None:
            bias = self.bias * self.bias_mask
        else:
            bias = self.bias
        return F.linear(input, weight, bias)

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        s += f'in_features={self.in_features}, '
        s += f'out_features={self.out_features}, '
        s += f'bias={self.bias is not None})'
        return s

# TODO inherit from nn.conv._ConvNd
class Conv2dMasked(nn.Module):

    def __init__(self, conv_layer, weight_mask, bias_mask=None):
        super(Conv2dMasked, self).__init__()

        self.weight = conv_layer.weight
        self.bias = conv_layer.bias

        # use register_buffer so model.to(device) works on fixed tensors like masks
        self.register_buffer("weight_mask", _ensure_tensor(weight_mask))
        self.register_buffer("bias_mask", _ensure_tensor(bias_mask))

        # Multiply weights by masks so metrics can count nonzeros
        self.weight_mask = _same_device(self.weight_mask, self.weight)
        self.weight.data.mul_(self.weight_mask)

        if bias_mask is not None:
            self.bias_mask = _same_device(self.bias_mask, self.bias)
            self.bias.data.mul_(self.bias_mask)
            assert self.bias is not None

        for attr in ['in_channels', 'out_channels', 'kernel_size', 'dilation',
                     'stride', 'padding', 'padding_mode', 'groups']:
            setattr(self, attr, getattr(conv_layer, attr))

    def forward(self, input):
        weight = self.weight * self.weight_mask
        if self.bias_mask is not None:
            bias = self.bias * self.bias_mask
        else:
            bias = self.bias

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
# TODO mask batchnorm for completion sake
# squeeze out Convs for channel pruning

masked_modules = {
    nn.Linear: LinearMasked,
    nn.Conv2d: Conv2dMasked,
}
