import argparse
import pathlib

import h5py
import numpy as np
import PIL.Image
from tqdm import tqdm


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = PIL.Image.open(f)
        return img.convert('RGB')


def binary_loader(path):

    with open(path, 'rb') as f:
        return np.frombuffer(f.read(), dtype='uint8')


def Imagefolder_to_hdf5(folder_path, output_file, binary=False):

    folder_path = pathlib.Path(folder_path)

    with h5py.File(output_file, 'w') as hf:

        hf.attrs['mode'] = 'binary' if binary else 'array'

        for label, class_path in enumerate(tqdm(sorted(folder_path.iterdir()))):
            class_name = class_path.stem
            class_grp = hf.create_group(class_name)

            for image_path in tqdm(sorted(class_path.iterdir()), desc=class_name, leave=False):
                image_name = image_path.stem
                if binary:
                    img_data = binary_loader(image_path)
                else:
                    img_data = pil_loader(image_path)
                class_grp.create_dataset(image_name, data=img_data)


parser = argparse.ArgumentParser(description='Convert a ImageFolder with dataset into HDF5')

parser.add_argument('-b', '--binary', dest='binary', action='store_true', default=False, help='Folder to crawl')
parser.add_argument('folder', type=str, help='Folder to crawl')
parser.add_argument('h5file', type=str, help='Output HDF5 filename')


if __name__ == '__main__':

    args = parser.parse_args()

    Imagefolder_to_hdf5(args.folder, args.h5file, args.binary)
