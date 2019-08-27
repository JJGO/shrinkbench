import os
import pathlib

WEIGHTS_DIR = pathlib.Path(os.path.abspath(__file__)).parent / 'pretrained'


def get_weights_path(root_path=None):
    if root_path is None:
        file_path = pathlib.Path(os.path.abspath(__file__))
        root_path = file_path.parent.parent / 'pretrained'

    paths = {
        'mnistnet': root_path / 'mnist',
        'cifar10_resnet': root_path / 'cifar10'
    }

    return paths


from .mnistnet import mnistnet
from .cifar10_resnet import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202