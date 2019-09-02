from torchvision import transforms, datasets

from . import dataset_path

normalize = transforms.Normalize((0.1307,), (0.3081,))
shape = (1, 28, 28)


def train_dataset(preproc=True, path=None):

    data_path = dataset_path('MNIST', path)

    train_preprocessing = None
    if preproc:
        train_preprocessing = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = datasets.MNIST(data_path,
                                   train=True,
                                   transform=train_preprocessing)
    return train_dataset


def val_dataset(preproc=True, path=None):

    data_path = dataset_path('MNIST', path)

    val_preprocessing = None
    if preproc:
        val_preprocessing = transforms.Compose([transforms.ToTensor(), normalize])

    val_dataset = datasets.MNIST(data_path,
                                 train=False,
                                 transform=val_preprocessing)
    return val_dataset
