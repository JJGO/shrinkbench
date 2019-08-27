from torchvision import transforms, datasets

from . import get_data_paths

normalize = transforms.Normalize((0.1307,), (0.3081,))
shape = (1, 28, 28)


def train_dataset(preproc=True):

    data_paths = get_data_paths()

    train_preprocessing = None
    if preproc:
        train_preprocessing = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = datasets.MNIST(data_paths['MNIST'],
                                   train=True,
                                   transform=train_preprocessing)
    return train_dataset


def val_dataset(preproc=True):

    data_paths = get_data_paths()

    val_preprocessing = None
    if preproc:
        val_preprocessing = transforms.Compose([transforms.ToTensor(), normalize])

    val_dataset = datasets.MNIST(data_paths['MNIST'],
                                 train=False,
                                 transform=val_preprocessing)
    return val_dataset
