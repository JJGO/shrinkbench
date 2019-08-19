import pathlib
import os


def get_data_paths(root_path=None):
    if root_path is None:
        file_path = pathlib.Path(os.path.abspath(__file__))
        root_path = file_path.parent.parent / 'data'

    paths = {
        'MNIST': root_path / 'MNIST',
        'ImageNet': root_path / 'ILSVRC2012_local',
    }

    # If not local, check remote
    if not paths['ImageNet'].exists():
        paths['ImageNet'] = root_path / 'ILSVRC2012'

    # Clear symlinks and check existence
    for data in paths:
        paths[data] = paths[data].resolve()
        assert paths[data].exists()

    return paths
