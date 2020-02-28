import pathlib

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

# Data is here : http://places2.csail.mit.edu/download.html
# We assume directory structure from
#   Small images (256 * 256) with easy directory structure


class Places365(Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None, **kwargs):
        root = pathlib.Path(root)
        root /= 'places365_standard'
        root /= 'train' if train else 'val'
        self.data = ImageFolder(root,
                                transform=transform,
                                target_transform=target_transform,
                                **kwargs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
