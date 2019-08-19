from torchvision import transforms, datasets

from . import get_data_paths

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def imagenet_train(preproc=True):

    data_paths = get_data_paths()

    train_preprocessing = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    if not preproc:
        train_preprocessing = None

    train_dataset = datasets.ImageNet(data_paths['ImageNet'],
                                      'train',
                                      transform=train_preprocessing)
    return train_dataset


def imagenet_val(preproc=True):

    data_paths = get_data_paths()

    val_preprocessing = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    if not preproc:
        val_preprocessing = None

    val_dataset = datasets.ImageNet(data_paths['ImageNet'],
                                    'val',
                                    transform=val_preprocessing)
    return val_dataset


def imagenet_datasets(preproc=True):
    return imagenet_train(preproc), imagenet_val(preproc)
