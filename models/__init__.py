import os
import pathlib


from .head import replace_head
from .mnistnet import MnistNet
from .cifar_resnet import (resnet20,
                           resnet32,
                           resnet44,
                           resnet56,
                           resnet110,
                           resnet1202)
from .cifar_resnet import (resnet20_100,
                           resnet32_100,
                           resnet44_100,
                           resnet56_100,
                           resnet110_100,
                           resnet1202_100)

from .cifar_vgg import vgg_bn_drop, vgg_bn_drop_100
