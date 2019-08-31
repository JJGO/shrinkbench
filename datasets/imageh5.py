import pathlib

import h5py
import PIL.Image
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


def make_dataset(hf):

    samples = []
    for label, class_ in enumerate(tqdm(sorted(hf), leave=False)):

        for image in sorted(hf[class_]):
            samples.append( (hf[class_][image].name, label) )

    return samples


def class_to_idx(hf):

    class_to_idx = {}
    for label, class_ in enumerate(sorted(hf)):
        class_to_idx[class_] = label

    return class_to_idx


def PIL_loader(array):
    return PIL.Image.fromarray(array)


class ImageHDF5Dataset(Dataset):

    def __init__(self, h5file, loader=PIL_loader,
                 transform=None,  target_transform=None):
        super(ImageHDF5Dataset, self).__init__()

        self.h5file = pathlib.Path(h5file)
        indexfile = h5file.with_suffix('.idx.npy')

        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        # Compile image & label list
        if indexfile.exists():
            self.samples = [(x, int(y)) for x, y in np.load(indexfile)]
        else:
            print("Compiling dataset indices")
            with h5py.File(self.h5file, 'r') as hf:
                self.samples = make_dataset(hf)
                np.save(indexfile, self.samples)
            # self.class_to_idx = class_to_idx(hf)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # File descriptor must be opened here because h5 does
        # support processing inheriting the file descriptor
        # https://gist.github.com/bkj/f448025fdef08c0609029489fa26ea2a
        path, target = self.samples[index]
        with h5py.File(self.h5file, 'r') as hf:
            sample = self.loader(hf[path].value)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


# class ImageHDF5Dataset(Dataset):

#     def __init__(self, h5file, loader=PIL_loader,
#                  transform=None,  target_transform=None):
#         super(ImageHDF5Dataset, self).__init__()

#         h5file = pathlib.Path(h5file)
#         indexfile = h5file.with_suffix('.idx.npy')

#         self.hf = h5py.File(h5file, 'r')
#         self.loader = loader
#         self.transform = transform
#         self.target_transform = target_transform
#         # Compile image & label list
#         if indexfile.exists():
#             self.samples = [(x, int(y)) for x, y in np.load(indexfile)]
#         else:
#             print("Compiling dataset indices")
#             self.samples = make_dataset(self.hf)
#             np.save(indexfile, self.samples)
#         self.class_to_idx = class_to_idx(self.hf)

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, index):
#         path, target = self.samples[index]
#         sample = self.loader(self.hf[path].value)
#         if self.transform is not None:
#             sample = self.transform(sample)
#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return sample, target

#     def __del__(self):
#         if hasattr(self, 'hf'):
#             self.hf.close()
