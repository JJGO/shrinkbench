import os
import pathlib

WEIGHTS_DIR = '../pretrained'


def weights_path(model, path=None):

    if path is None:
        path = WEIGHTS_DIR
        # Look for the dataset in known paths
        if 'WEIGHTSPATH' in os.environ:
            path = os.environ['WEIGHTSPATH'] + ':' + path
    paths = [pathlib.Path(p) for p in path.split(':')]

    for p in paths:
        for root, dirs, files in os.walk(p, followlinks=True):
            if model in files:
                return pathlib.Path(root) / model
            # for file in files:
            #     path = pathlib.Path(root) / file
            #     if str(path).endswith(model):
            #         return path
    else:
        raise LookupError(f"Could not find {model} in {paths}")


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
