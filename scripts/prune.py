import argparse
import json
import os
import time


def jsonfile(file):
    with open(file, 'r') as f:
        s = json.load(f)
    return s


# def float_fraction(x):
#     from fractions import Fraction
#     return float(Fraction(x))


parser = argparse.ArgumentParser(description='Train a [pruned] Vision Net and finetune it')

parser.add_argument('-s', '--strategy', dest='strategy', type=str, help='Pruning strategy', default=None)
parser.add_argument('-c', '--compression', dest='compression', type=int, help='Pruning Strategy', default=1)
parser.add_argument('-d', '--dataset', dest='dataset', type=str, help='Dataset to train on')
parser.add_argument('-m', '--model', dest='model', type=str, help='What CNN to use')
# parser.add_argument('-w', '--weights', dest='weights', action='store_true', default=True, help='Use pretrained weights if possible')
parser.add_argument('-S', '--seed', dest='seed', type=int, help='Random seed for reproducibility', default=42)
parser.add_argument('-P', '--path', dest='path', type=str, help='path to save', default=None)
parser.add_argument('-r', '--resume', dest='resume', type=str, help='Checkpoint to resume from', default=None)
parser.add_argument('--resume-optim', dest='resume_optim', action='store_true', default=False, help='Resume also optim')
parser.add_argument('-n', '--debug', dest='debug', action='store_true', default=False, help='Enable debug mode for logging')
parser.add_argument('-D', '--dl', dest='dl_kwargs', type=json.loads, help='JSON string of DataLoader parameters', default=tuple())
parser.add_argument('-T', '--train', dest='train_kwargs', type=json.loads, help='JSON string of Train parameters', default=tuple())
parser.add_argument('-g', dest='gpuid', type=str, help='GPU id', default="0")
parser.add_argument('--no-pretrained', dest='pretrained', action='store_false', default=True, help='Do not use pretrained model')
parser.add_argument('-l', '--retain', dest='retain', action='store_true', default=False, help='Do not release GPU')

if __name__ == '__main__':

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

    from flor.experiment import PruningExperiment

    exp = PruningExperiment(dataset=args.dataset,
                            model=args.model,
                            strategy=args.strategy,
                            compression=args.compression,
                            seed=args.seed,
                            path=args.path,
                            pretrained=args.pretrained,
                            dl_kwargs=args.dl_kwargs,
                            train_kwargs=args.train_kwargs,
                            debug=args.debug,
                            resume=args.resume,
                            resume_optim=args.resume_optim)

    exp.run()

    # TODO - parse signals
    # signal.signal(signal.SIGINT, self.SIGINT_handler)
    # signal.signal(signal.SIGQUIT, self.SIGQUIT_handler)

    while args.retain:
        time.sleep(60)
