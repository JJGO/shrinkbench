import argparse
import csv
import time
import timeit
import numpy as np

parser = argparse.ArgumentParser(description='Profile ImageNet Dataloaders')

parser.add_argument('-o', '--output', dest='outfile', type=str, help='Output CSV')
parser.add_argument('-n', '--iter', dest='iter', type=int, help='Number of dataloader iters', default=10)
parser.add_argument('-r', '--repeat', dest='repeat', type=int, help='How many times to run whole code', default=7)
parser.add_argument('split', type=str, help='train|val')
parser.add_argument('loader', type=str, help='folder|hdf5')
parser.add_argument('location', type=str, help='local|remote')
parser.add_argument('num_workers', type=int, help='number of threads 0,1,2,...')
parser.add_argument('batch_size', type=int, help='batch size')

columns = ['split', 'loader', 'location', 'num_workers', 'batch_size']
columns += ['mean', 'std', 'setup']


def imagenet_dataloader(split, loader, location, num_workers, batch_size):
    paths = {
        'local': '/home/neuro/datasets',
        'remote': '../data',
    }
    from torch.utils.data import DataLoader
    if loader == 'hdf5':
        from flor.datasets.imageneth5 import train_dataset, val_dataset
    else:
        from flor.datasets.imagenet import train_dataset, val_dataset
    if split == 'train':
        data = train_dataset(preproc=True, path=paths[location])
    else:
        data = val_dataset(preproc=True, path=paths[location])

    dl_kwargs = {'shuffle': split == 'train',
                 'batch_size': batch_size,
                 'pin_memory': True,
                 'num_workers': num_workers}
    dl = iter(DataLoader(data, **dl_kwargs))
    next(dl)
    return dl


if __name__ == '__main__':

    args = parser.parse_args()
    since = time.time()
    dl = imagenet_dataloader(args.split, args.loader, args.location, args.num_workers, args.batch_size)
    setup_time = time.time() - since
#     SETUP_CODE = f"""
# from __main__ import imagenet_dataloader
# dl = imagenet_dataloader({args.split}, {args.loader}, {args.location}, {args.num_workers}, {args.batch_size})"""

    TEST_CODE = """next(dl)"""

    timings = timeit.repeat(stmt=TEST_CODE, globals={"dl": dl}, repeat=args.repeat, number=args.iter)

    row = [args.split, args.loader, args.location, args.num_workers, args.batch_size]
    timings = np.array(timings) / args.iter
    mean = np.mean(timings)
    std = np.std(timings)
    print(*row)
    row += [mean, std, setup_time]
    print(f"{mean*1000:.2f} ms ± {std*1000:.2f} ms per loop (mean ± std of {args.repeat} runs {args.iter} loops each)")

    with open('imagenet_profiling_results.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)
