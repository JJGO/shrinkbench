import os
import pathlib

WEIGHTS_DIR = pathlib.Path(os.path.abspath(__file__)).parent / 'pretrained'


def get_weights_path(root_path=None):
    if root_path is None:
        file_path = pathlib.Path(os.path.abspath(__file__))
        root_path = file_path.parent.parent / 'pretrained'

    paths = {
        'mnistnet': root_path / 'mnistnet.pt',
    }

    return paths


from .mnistnet import mnistnet
