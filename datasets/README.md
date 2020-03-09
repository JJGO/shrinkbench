Datasets
========

Supported datasets are as follows:

- MNIST (for debugging purposes only)
- CIFAR10
- CIFAR100
- ImageNet
- Places365

The script expects data to be passed in a `path` variable or thorugh a `DATAPATH` environment variables. In both cases, the variable is parsed as a semicolon separated string.

For instance the path variable `/path/to/small:/other/path/to/large` would be able to find these datasets if they follow the following directory structure.

```
/path/to/small
├── MNIST
│   └── MNIST
├── CIFAR10
│   ├── cifar-10-batches-py
│   └── cifar-10-python.tar.gz
└── CIFAR100
    ├── cifar-100-python
    └── cifar-100-python.tar.gz

/other/path/to/large
├── ILSVRC2012
│   ├── ILSVRC2012_img_train
│   └── ILSVRC2012_img_val
└── Places365
    └── places365_standard
        └── train
        └── val
```

The provided datasets include proper channel normalization and the transforms that pretrained models use for their pipelines.

For small datasets (MNIST, CIFAR10, CIFAR100), it is easiest to download through [`torchvision.datasets`](https://pytorch.org/docs/stable/torchvision/datasets.html) by setting `download=True`.

Data for ImageNet ILSVRC dataset can be obtained from the [official website](http://image-net.org/challenges/LSVRC/2012/) or from Academic Torrents ([train](http://academictorrents.com/details/a306397ccf9c2ead27155983c254227c0fd938e2), [val](http://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5)).

Data for Places365 data can be downloaded from the [official website](http://places2.csail.mit.edu/download.html). We recommend the "Small images (256 * 256) with easy directory structure" version for easy interoperability with the provided dataloader (which is based on torchvision's `ImageFolder` dataloader).