# # https://github.com/CSAILVision/places365/blob/master/train_placesCNN.py
# from torchvision import transforms, datasets

# from . import get_data_paths


# # TODO : Bolei used the normalization for Imagenet, not the one for Places!
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])


# # TODO : Use HDF5 files instead


# def train_dataset(preproc=True):

#     data_paths = get_data_paths()

#     train_preprocessing = None
#     if preproc:
#         train_preprocessing = transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize,
#         ])
#     train_dataset = datasets.ImageFolder(data_paths['Places365'] / 'train',
#                                          transform=train_preprocessing)
#     return train_dataset


# def val_dataset(preproc=True):

#     data_paths = get_data_paths()

#     val_preprocessing = None
#     if preproc:
#         val_preprocessing = transforms.Compose([
#             transforms.Resize(256), # TODO Necessary? Isn't data 256?
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             normalize,
#         ])

#     val_dataset = datasets.ImageFolder(data_paths['Places365'] / 'val',
#                                        transform=val_preprocessing)
#     return val_dataset
