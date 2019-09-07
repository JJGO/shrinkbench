import json
import pathlib
import string

import pandas as pd
import matplotlib.pyplot as plt

COLUMNS = ['dataset', 'model', 'strategy', 'compression',
           'pre_acc1', 'pre_acc5', 'post_acc1', 'post_acc5',
           'size', 'size_nz', 'real_compression',
           'memory', 'memory_nz',
           'flops', 'flops_nz', 'speedup',
           'completed', 'seed',
           'batch_size', 'epochs', 'optim', 'lr',
           'resume',
           'path']
CMAP = plt.get_cmap('Set1')


class AutoMap:

    def __init__(self, objects):
        self.mapping = {}
        self.idx = 0
        self.objects = objects

    def __getitem__(self, key):
        if key not in self.mapping:
            self.mapping[key] = self.objects[self.idx]
            self.idx += 1
        return self.mapping[key]

    def __repr__(self):
        return repr(self.mapping)


colors = AutoMap(CMAP.colors)


class AutoMap:

    def __init__(self, objects):
        self.mapping = {}
        self.idx = 0
        self.objects = objects

    def __getitem__(self, key):
        if key not in self.mapping:
            self.mapping[key] = self.objects[self.idx]
            self.idx += 1
        return self.mapping[key]

    def __repr__(self):
        return repr(self.mapping)


def df_from_results(results_path, glob='*'):
    results = []
    results_path = pathlib.Path(results_path)
    for exp in results_path.glob(glob):
        if not (exp / 'params.json').exists():
            continue
        with open(exp / 'params.json', 'r') as f:
            params = json.load(f)
        if (exp / 'metrics.json').exists():
            with open(exp / 'metrics.json', 'r') as f:
                metrics = json.load(f)
            row = []
            # dataset, model
            row += [eval(params['dataset']), eval(params['model'])]
            # strategy, compression
            row += [eval(params['strategy']), eval(params['compression'])]
            # pre-finetuning val acc {top1, top5}
            row += [metrics['pre']['val_acc1'], metrics['pre']['val_acc5']]
            # best finetuning val acc {top1, top5}
            if params['train_kwargs']['epochs'] > 0:
                finetuning = pd.read_csv(exp / 'finetuning.csv')
                if eval(params['dataset']).startswith('ImageNet'):
                    # TODO remove this constraint
                    finetuning = finetuning[finetuning['epoch'] <= 20]
                row += [finetuning['val_acc1'].max(), finetuning['val_acc5'].max()]
            else:
                assert 'post' in metrics
                row += [metrics['post']['val_acc1'], metrics['post']['val_acc5']]

            completed = 'post' in metrics
            metrics = metrics['post'] if 'post' in metrics else metrics['pre']
            # size, nonzero size, achieved compression
            row += [metrics['size'], metrics['size_nz'], metrics['compression_ratio']]
            # memory
            row += [metrics['memory'], metrics['memory_nz']]
            # flops
            row += [metrics['flops'], metrics['flops_nz'], metrics['flops']/metrics['flops_nz']]
            # completed, Random Seed, epochs, experiment path
            row += [completed, eval(params['seed'])]
            row += [params['dl_kwargs']['batch_size'],
                    params['train_kwargs']['epochs'],
                    params['train_kwargs']['optim'],
                    params['train_kwargs']['lr']]
            row += [None if 'resume' not in params else eval(params['resume'])]
            row += [str(exp)]
            results.append(row)

    return pd.DataFrame(data=results, columns=COLUMNS)


def plot_acc_compression_strategy(df, strat, top=1, pre=False, prefix="", diff=False):
    global colors
    strat_df = df[(df['strategy'] == strat) | df['strategy'].isna()].sort_values('compression', ascending=False)
    strat_df = strat_df.drop(columns=['completed'])
    mean = strat_df.groupby('compression', as_index=False).mean()
    std = strat_df.groupby('compression', as_index=False).std()
    strat_nick = "".join(c for c in strat if c not in string.ascii_lowercase)  # strat[:4]
    pre_acc = mean[f"pre_acc{top}"]
    post_acc = mean[f"post_acc{top}"]
    if diff:
        pre_acc = -(pre_acc - pre_acc[0])
        post_acc = -(post_acc - post_acc[0])
    if pre:
        plt.plot(mean['real_compression'], pre_acc, ls='--', marker='.',
                 label=f"{prefix}{strat_nick} Pre", color=colors[strat_nick])
    plt.errorbar(mean['real_compression'], post_acc, ls='-', marker='.',
                 label=f"{prefix}{strat_nick} Post", color=colors[strat_nick], yerr=std[f"post_acc{top}"])


def plot_acc_compression(df, base_acc=0.1, top=1, pre=False, fig=True, prefix="", delta=False):
    global color_idx
    if fig:
        # color_idx = 0
        plt.figure(figsize=(15, 8))

    if not delta:
        plt.axhline(base_acc, ls='-.', color='gray')

    assert len(set(df['dataset'])) == 1, "More than one dataset in datataframe"
    assert len(set(df['model'])) == 1, "More than one model in datataframe"

    strategies = [s for s in set(df['strategy']) if s is not None]

    for s in sorted(strategies):
        plot_acc_compression_strategy(df, s, top, pre, prefix, delta)
        # color_idx += 1

    plt.ylabel('Accuracy' if not delta else 'Decrease in Accuracy')
    plt.xlabel('Compression')
    plt.xscale('log')
    ticks = sorted(set(df['compression']))
    plt.xticks(ticks, map(str, ticks))
    plt.legend()

    dataset, model, *_ = df.iloc[0]
    dataset, model
    plt.title(f" Accuracy vs. Compression - {dataset} + {model}")


def plot_error_compression_strategy(df, strat, i, prefix="", ls='-'):
    strat_df = df[df['strategy'] == strat].sort_values('compression', ascending=False)
    strat_df = strat_df.drop(columns=['completed'])
    mean = strat_df.groupby('compression', as_index=False).mean()
    # std = strat_df.groupby('compression', as_index=False).std()
    desired_compression = mean['compression']
    strat_nick = "".join(c for c in strat if c in string.ascii_uppercase)  # strat[:4]
    relative_error = (mean['real_compression']-desired_compression)/desired_compression
    plt.plot(desired_compression, relative_error*100, ls=ls, marker='o',
             color=CMAP.colors[i], label=f"{prefix}{strat_nick}")


def plot_error_compression(df, fig=True, prefix="", ls='-'):
    if fig:
        plt.figure(figsize=(15, 8))
    assert len(set(df['dataset'])) == 1, "More than one dataset in datataframe"
    assert len(set(df['model'])) == 1, "More than one model in datataframe"

    strategies = [s for s in set(df['strategy']) if s is not None]

    for i, s in enumerate(strategies):
        plot_error_compression_strategy(df, s, i, prefix, ls)

    plt.xscale('log')
    ticks = [c for c in sorted(set(df['compression'])) if c > 1]
    plt.xticks(ticks, map(str, ticks))
    plt.ylabel('Relative Error [%]')
    plt.xlabel('Desired Compression')
    plt.legend()
    dataset, model, *_ = df.iloc[0]
    dataset, model
    plt.title(f"Relative Compression Error - {dataset} + {model}")
