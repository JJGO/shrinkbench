# https://github.com/pytorch/examples/blob/master/imagenet/main.py
from torchvision import transforms, datasets

from . import dataset_path

import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
shape = (3, 224, 224)


def train_dataset(preproc=True, path=None):

    data_path = dataset_path('ILSVRC2012', path)

    train_preprocessing = None
    if preproc:
        train_preprocessing = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = datasets.ImageNet(data_path,
                                      'train',
                                      transform=train_preprocessing)
    return train_dataset


def val_dataset(preproc=True, path=None):

    data_path = dataset_path('ILSVRC2012', path)

    val_preprocessing = None
    if preproc:
        val_preprocessing = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    val_dataset = datasets.ImageNet(data_path,
                                    'val',
                                    transform=val_preprocessing)
    return val_dataset
