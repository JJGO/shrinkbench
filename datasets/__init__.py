import pathlib
import os


def get_data_paths(root_path=None):
    if root_path is None:
        file_path = pathlib.Path(os.path.abspath(__file__))
        root_path = file_path.parent.parent / 'data'

    paths = {
        'MNIST': root_path,
        'ImageNet': root_path / 'ILSVRC2012_local',
        'ImageNetHDF5': root_path / 'ILSVRC2012_HDF5_local',
        'CIFAR10': root_path / 'CIFAR10',
        'CIFAR100': root_path / 'CIFAR100',
        # 'Places365': root_path / 'Places365'
    }

    # Imagenet{,_HDF5}, check if local, else remote
    if not paths['ImageNet'].exists():
        paths['ImageNet'] = root_path / 'ILSVRC2012'
    if not paths['ImageNetHDF5'].exists():
        paths['ImageNetHDF5'] = root_path / 'ILSVRC2012_HDF5'

    # Clear symlinks and check existence
    for data in paths:
        paths[data] = paths[data].resolve()
        assert paths[data].exists()

    return paths


def get_datasets(dataset, preproc=True):

    if dataset == 'MNIST':
        from .mnist import train_dataset, val_dataset
    elif dataset == 'ImageNet':
        from .imagenet import train_dataset, val_dataset
    elif dataset == 'ImageNetHDF5':
        from .imageneth5 import train_dataset, val_dataset
    elif dataset == 'CIFAR10':
        from .cifar10 import train_dataset, val_dataset
    elif dataset == 'CIFAR100':
        from .cifar100 import train_dataset, val_dataset
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    return train_dataset(preproc), val_dataset(preproc)
