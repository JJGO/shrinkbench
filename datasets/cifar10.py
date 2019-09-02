from torchvision import transforms, datasets

from . import dataset_path

# TODO this is Imagenet norm values
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
shape = (3, 32, 32)


def train_dataset(preproc=True, path=None):

    data_path = dataset_path('CIFAR10', path)

    train_preprocessing = None
    if preproc:
        train_preprocessing = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = datasets.CIFAR10(data_path,
                                     train=True,
                                     transform=train_preprocessing)
    return train_dataset


def val_dataset(preproc=True, path=None):

    data_path = dataset_path('CIFAR10', path)

    val_preprocessing = None
    if preproc:
        val_preprocessing = transforms.Compose([transforms.ToTensor(), normalize])

    val_dataset = datasets.CIFAR10(data_path,
                                   train=False,
                                   transform=val_preprocessing)
    return val_dataset
