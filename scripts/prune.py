import argparse
import json
import os
import time


def jsonfile(file):
    with open(file, 'r') as f:
        s = json.load(f)
    return s


parser = argparse.ArgumentParser(description='Train a [pruned] Vision Net and finetune it')

parser.add_argument('-s', '--strategy', dest='strategy', type=str, help='Pruning strategy')
parser.add_argument('-p', '--pruning', dest='pruning', type=float, help='')
parser.add_argument('-d', '--dataset', dest='dataset', type=str, help='Dataset to train on')
parser.add_argument('-m', '--model', dest='model', type=str, help='What CNN to use')
parser.add_argument('-r', '--seed', dest='seed', type=int, help='Random seed for reproducibility', default=42)
parser.add_argument('-P', '--path', dest='path', type=str, help='path to save', default=None)
parser.add_argument('-D', '--dl', dest='dl_kwargs', type=json.loads, help='JSON string of DataLoader parameters', default=tuple())
parser.add_argument('-T', '--train', dest='train_kwargs', type=json.loads, help='JSON string of Train parameters', default=tuple())
parser.add_argument('-g', dest='gpuid', type=str, help='GPU id', default="0")
parser.add_argument('-l', '--retain', dest='retain', action='store_true', default=False, help='Do not release GPU')

if __name__ == '__main__':

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

    from flor.experiment import PruningExperiment

    exp = PruningExperiment(strategy=args.strategy,
                            pruning=args.pruning,
                            dataset=args.dataset,
                            model=args.model,
                            seed=args.seed,
                            path=args.path,
                            dl_kwargs=args.dl_kwargs,
                            train_kwargs=args.train_kwargs)

    exp.run()

    # TODO - parse signals
    # signal.signal(signal.SIGINT, self.SIGINT_handler)
    # signal.signal(signal.SIGQUIT, self.SIGQUIT_handler)

    while args.retain:
        time.sleep(60)
