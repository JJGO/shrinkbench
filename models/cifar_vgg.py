# Faithful reimplementation of
# https://github.com/szagoruyko/cifar.torch/blob/master/models/vgg_bn_drop.lua
# BLOG POST : http://torch.ch/blog/2015/07/30/cifar.html

import math
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.nn.init as init

from . import weights_path


class ConvBNReLU(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(ConvBNReLU, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=3//2)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class VGGBnDrop(nn.Module):

    def __init__(self, num_classes=10):

        super(VGGBnDrop, self).__init__()

        self.num_classes = num_classes

        self.features = nn.Sequential(

            ConvBNReLU(3, 64), nn.Dropout(0.3),
            ConvBNReLU(64, 64),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            ConvBNReLU(64, 128), nn.Dropout(0.4),
            ConvBNReLU(128, 128),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            ConvBNReLU(128, 256), nn.Dropout(0.4),
            ConvBNReLU(256, 256), nn.Dropout(0.4),
            ConvBNReLU(256, 256),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            ConvBNReLU(256, 512), nn.Dropout(0.4),
            ConvBNReLU(512, 512), nn.Dropout(0.4),
            ConvBNReLU(512, 512),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            ConvBNReLU(512, 512), nn.Dropout(0.4),
            ConvBNReLU(512, 512), nn.Dropout(0.4),
            ConvBNReLU(512, 512),
            nn.MaxPool2d(2, 2, ceil_mode=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes),
        )

    def forward(self, input):

        x = self.features(input)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def reset_weights(self):

        def init_weights(module):
            if isinstance(module, nn.Conv2d):
                fan_in, _ = init._calculate_fan_in_and_fan_out(module.weight)
                init.normal_(module.weight, 0, math.sqrt(2)/fan_in)
                init.zeros_(module.bias)

        self.apply(init_weights)


def vgg_bn_drop(pretrained=True):
    model = VGGBnDrop(num_classes=10)
    if pretrained:
        weights = weights_path('vgg_bn_drop.pt')
        model.load_state_dict(torch.load(weights))
    # else:
        # model.reset_weights()
    return model


def vgg_bn_drop_100(pretrained=True):
    model = VGGBnDrop(num_classes=100)
    if pretrained:
        weights = weights_path('vgg_bn_drop_100.pt')
        model.load_state_dict(torch.load(weights))
    # else:
        # model.reset_weights()
    return model
