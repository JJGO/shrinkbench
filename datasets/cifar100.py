from torchvision import transforms, datasets

from . import get_data_paths

# TODO this is Imagenet norm values
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
shape = (3, 32, 32)


def train_dataset(preproc=True):

    data_paths = get_data_paths()

    train_preprocessing = None
    if preproc:
        train_preprocessing = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = datasets.CIFAR100(data_paths['CIFAR100'],
                                      train=True,
                                      transform=train_preprocessing)
    return train_dataset


def val_dataset(preproc=True):

    data_paths = get_data_paths()

    val_preprocessing = None
    if preproc:
        val_preprocessing = transforms.Compose([transforms.ToTensor(), normalize])

    val_dataset = datasets.CIFAR100(data_paths['CIFAR100'],
                                    train=False,
                                    transform=val_preprocessing)
    return val_dataset
