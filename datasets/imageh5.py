import h5py
import PIL.Image
from torch.utils.data import Dataset


def make_dataset(hf):

    samples = []
    class_to_idx = {}

    for label, class_ in enumerate(sorted(hf)):

        class_to_idx[class_] = label

        for image in sorted(hf[class_]):
            samples.append( (f"{class_}/{image}", label) )

    return samples, class_to_idx


def PIL_loader(array):
    return PIL.Image.fromarray(array)


class ImageHDF5Dataset(Dataset):

    def __init__(self, h5file, loader=PIL_loader,
                 transform=None,  target_transform=None):
        super(ImageHDF5Dataset, self).__init__()

        self.hf = h5py.File(h5file, 'r')
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        # Compile image & label list
        self.samples, self.class_to_idx = make_dataset(self.hf)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(self.hf[path].value)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __del__(self):
        self.hf.close()
