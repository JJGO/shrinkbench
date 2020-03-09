import pathlib
import os

from torchvision import transforms, datasets

from . import places365

_constructors = {
    'MNIST': datasets.MNIST,
    'CIFAR10': datasets.CIFAR10,
    'CIFAR100': datasets.CIFAR100,
    'ImageNet': datasets.ImageNet,
    'Places365': places365.Places365
}


def dataset_path(dataset, path=None):
    """Get the path to a specified dataset

    Arguments:
        dataset {str} -- One of MNIST, CIFAR10, CIFAR100, ImageNet, Places365

    Keyword Arguments:
        path {str} -- Semicolon separated list of paths to look for dataset folders (default: {None})

    Returns:
        dataset_path -- pathlib.Path for the first match

    Raises:
        ValueError -- If no path is provided and DATAPATH is not set
        LookupError -- If the given dataset cannot be found
    """
    if path is None:
        # Look for the dataset in known paths
        if 'DATAPATH' in os.environ:
            path = os.environ['DATAPATH']
            paths = [pathlib.Path(p) for p in path.split(':')]
        else:
            raise ValueError(f"No path specified. A path must be provided, \n \
                           or the folder must be listed in your DATAPATH")

    paths = [pathlib.Path(p) for p in path.split(':')]

    for p in paths:
        p = (p / dataset).resolve()
        if p.exists():
            print(f"Found {dataset} under {p}")
            return p
    else:
        raise LookupError(f"Could not find {dataset} in {paths}")


def dataset_builder(dataset, train=True, normalize=None, preproc=None, path=None):
    """Build a torch.utils.Dataset with proper preprocessing

    Arguments:
        dataset {str} -- One of MNIST, CIFAR10, CIFAR100, ImageNet, Places365

    Keyword Arguments:
        train {bool} -- Whether to return train or validation set (default: {True})
        normalize {torchvision.Transform} -- Transform to normalize data channel wise (default: {None})
        preproc {list(torchvision.Transform)} -- List of preprocessing operations (default: {None})
        path {str} -- Semicolon separated list of paths to look for dataset folders (default: {None})

    Returns:
        torch.utils.data.Dataset -- Dataset object with transforms and normalization
    """
    if preproc is not None:
        preproc += [transforms.ToTensor()]
        if normalize is not None:
            preproc += [normalize]
        preproc = transforms.Compose(preproc)

    kwargs = {'transform': preproc}
    if dataset == 'ImageNet':
        kwargs['split'] = 'train' if train else 'val'
    else:
        kwargs['train'] = train

    path = dataset_path(dataset, path)

    return _constructors[dataset](path, **kwargs)


def MNIST(train=True, path=None):
    """Thin wrapper around torchvision.datasets.CIFAR10
    """
    mean, std = 0.1307, 0.3081
    normalize = transforms.Normalize(mean=(mean,), std=(std,))
    dataset = dataset_builder('MNIST', train, normalize, [], path)
    dataset.shape = (1, 28, 28)
    return dataset


def CIFAR10(train=True, path=None):
    """Thin wrapper around torchvision.datasets.CIFAR10
    """
    mean, std = [0.491, 0.482, 0.447], [0.247, 0.243, 0.262]
    normalize = transforms.Normalize(mean=mean, std=std)
    if train:
        preproc = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4)]
    else:
        preproc = []
    dataset = dataset_builder('CIFAR10', train, normalize, preproc, path)
    dataset.shape = (3, 32, 32)
    return dataset


def CIFAR100(train=True, path=None):
    """Thin wrapper around torchvision.datasets.CIFAR100
    """
    mean, std = [0.507, 0.487, 0.441], [0.267, 0.256, 0.276]
    normalize = transforms.Normalize(mean=mean, std=std)
    if train:
        preproc = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4)]
    else:
        preproc = []
    dataset = dataset_builder('CIFAR100', train, normalize, preproc, path)
    dataset.shape = (3, 32, 32)
    return dataset


def ImageNet(train=True, path=None):
    """Thin wrapper around torchvision.datasets.ImageNet
    """
    # ImageNet loading from files can produce benign EXIF errors
    import warnings
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    if train:
        preproc = [transforms.RandomResizedCrop(224),
                   transforms.RandomHorizontalFlip()]
    else:
        preproc = [transforms.Resize(256), transforms.CenterCrop(224)]
    dataset = dataset_builder('ImageNet', train, normalize, preproc, path)
    dataset.shape = (3, 224, 224)
    return dataset


def Places365(train=True, path=None):
    """Thin wrapper around .datasets.places365.Places365
    """

    # Note : Bolei used the normalization for Imagenet, not the one for Places!
    # # https://github.com/CSAILVision/places365/blob/master/train_placesCNN.py
    # So these are kept so weights are compatible
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    normalize = transforms.Normalize((mean,), (std,))
    if train:
        preproc = [transforms.RandomResizedCrop(224),
                   transforms.RandomHorizontalFlip()]
    else:
        preproc = [transforms.Resize(256), transforms.CenterCrop(224)]
    dataset = dataset_builder('Places365', train, normalize, preproc, path)
    dataset.shape = (3, 224, 224)
    return dataset
