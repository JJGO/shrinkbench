import pathlib
import os

DATAPATH = '/home/neuro/datasets/:../data'


def dataset_path(dataset, path=None):

    if path is None:
        path = DATAPATH
        # Look for the dataset in known paths
        if 'DATAPATH' in os.environ:
            path = os.environ['DATAPATH'] + ':' + path
    paths = [pathlib.Path(p) for p in path.split(':')]

    for p in paths:
        p = (p / dataset).resolve()
        if p.exists():
            print(f"Found {dataset} under {p}")
            return p
    else:
        raise LookupError(f"Could not find {dataset} in {paths}")


def train_val_datasets(dataset, preproc=True, path=None):

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
    elif dataset == 'zeros224':
        from .zeros224 import train_dataset, val_dataset
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    return train_dataset(preproc, path), val_dataset(preproc, path)
