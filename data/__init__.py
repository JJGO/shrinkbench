import pathlib
import os


def data_paths(root_path=None):
    if root_path is None:
        root_path = pathlib.Path(os.path.abspath(__file__)).parent

    paths = {
        'MNIST': root_path / 'MNIST',
        'ImageNet': root_path / 'ILSVRC2012_localJJ',
    }

    # If not local, check remote
    if not paths['ImageNet'].exists():
        paths['ImageNet'] = root_path / 'ILSVRC2012'

    # Clear symlinks and check existence
    for data in paths:
        paths[data] = paths[data].resolve()
        assert paths[data].exists()

    return paths
