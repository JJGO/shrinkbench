import json
import pathlib

import pandas as pd
import matplotlib.pyplot as plt

COLUMNS = ['dataset', 'model', 'strategy', 'compression',
           'pre_acc1', 'pre_acc5', 'post_acc1', 'post_acc5',
           'size', 'size_nz', 'real_compression',
           'memory', 'memory_nz',
           'flops', 'flops_nz', 'speedup',
           'completed', 'seed', 'epochs', 'path']
# STRATEGIES = ['MagnitudePruning', 'ChannelPruning', 'RandomPruning']
CMAP = plt.get_cmap('Set1')

color_idx = 0


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
            row += [completed, params['seed'], params['train_kwargs']['epochs'], str(exp)]

            results.append(row)
    return pd.DataFrame(data=results, columns=COLUMNS)


def plot_acc_compression_strategy(df, strat, i, top=1, pre=False, prefix=""):
    strat_df = df[(df['strategy'] == strat) | df['strategy'].isna()].sort_values('compression', ascending=False)
    strat_df = strat_df.drop(columns=['completed'])
    mean = strat_df.groupby('compression', as_index=False).mean()
    std = strat_df.groupby('compression', as_index=False).std()

    if pre:
        plt.plot(mean['real_compression'], mean[f"pre_acc{top}"], ls='--', marker='.',
                 label=f"{prefix}{strat[:4]} Pre", color=CMAP.colors[i])
    plt.errorbar(mean['real_compression'], mean[f"post_acc{top}"], ls='-', marker='.',
                 label=f"{prefix}{strat[:4]} Post", color=CMAP.colors[i], yerr=std[f"post_acc{top}"])


def plot_acc_compression(df, base_acc=0.1, top=1, pre=False, fig=True, prefix=""):
    global color_idx
    if fig:
        color_idx = 0
        plt.figure(figsize=(15, 8))

    plt.axhline(base_acc, ls='-.', color='gray')

    assert len(set(df['dataset'])) == 1, "More than one dataset in datataframe"
    assert len(set(df['model'])) == 1, "More than one model in datataframe"

    strategies = [s for s in set(df['strategy']) if s is not None]

    for s in sorted(strategies):
        plot_acc_compression_strategy(df, s, color_idx, top, pre, prefix)
        color_idx += 1

    plt.ylabel('Accuracy')
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
    relative_error = (mean['real_compression']-desired_compression)/desired_compression
    plt.plot(desired_compression, relative_error*100, ls=ls, marker='o',
             color=CMAP.colors[i], label=f"{prefix}{strat[:5]}")


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
