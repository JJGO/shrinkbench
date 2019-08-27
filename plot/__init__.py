import json
import pathlib

import pandas as pd
import matplotlib.pyplot as plt

COLUMNS = ['dataset', 'model', 'strategy', 'pruning',
           'pre_acc1', 'pre_acc5', 'post_acc1', 'post_acc5',
           'size', 'size_nz', 'compression', 'memory', 'memory_nz', 'flops', 'flops_nz',
           'seed']
STRATEGIES = ['MagnitudePruning', 'ChannelPruning', 'RandomPruning']
CMAP = plt.get_cmap('Set1')


def df_from_results(results_path, glob='*'):
    results = []
    results_path = pathlib.Path(results_path)
    for exp in results_path.glob(glob):
        with open(exp / 'params.json', 'r') as f:
            params = json.load(f)
        with open(exp / 'metrics.json', 'r') as f:
            metrics = json.load(f)
        if 'post' in metrics:
            row = []
            row += [eval(params['dataset']), eval(params['model'])]
            row += [eval(params['strategy']), eval(params['pruning'])]
            row += [metrics['pre']['val_acc1'], metrics['pre']['val_acc5']]
#             row += [metrics['post']['val_acc1'], metrics['post']['val_acc5']]
            finetuning = pd.read_csv(exp / 'finetuning.csv')
            row += [finetuning['val_acc1'].max(), finetuning['val_acc5'].max()]
            row += [metrics['post']['size'], metrics['post']['size_nz'], metrics['post']['compression_ratio']]
            row += [metrics['post']['memory'], metrics['post']['memory_nz']]
            row += [metrics['post']['flops'], metrics['post']['flops_nz']]
            row += [params['seed']]
            results.append(row)
    return pd.DataFrame(data=results, columns=COLUMNS)


def plot_acc_compression_strategy(df, strat, i, top=1):
    strat_df = df[df['strategy'] == strat].sort_values('pruning', ascending=False)
    mean = strat_df.groupby('pruning', as_index=False).mean()
    std = strat_df.groupby('pruning', as_index=False).std()

    plt.plot(mean['compression'], mean[f"pre_acc{top}"], ls='--', marker='o',
             label=f"{strat[:4]} Pre", color=CMAP.colors[i])
    plt.errorbar(mean['compression'], mean[f"post_acc{top}"], ls='-', marker='o',
                 label=f"{strat[:4]} Post", color=CMAP.colors[i], yerr=std[f"post_acc{top}"])


def plot_acc_compression(df, base_acc=0.1, top=1):
    plt.figure(figsize=(15, 8))

    plt.axhline(base_acc, ls='-.', color='gray')

    assert len(set(df['dataset'])) == 1, "More than one dataset in datataframe"
    assert len(set(df['model'])) == 1, "More than one model in datataframe"

    for i, s in enumerate(STRATEGIES):
        plot_acc_compression_strategy(df, s, i, top)

    plt.ylabel('Accuracy')
    plt.xlabel('Compression')
    plt.xscale('log')
    ticks = sorted(set(1/df['pruning']))
    plt.xticks(ticks, map(str, ticks))
    plt.legend()

    dataset, model, *_ = df.iloc[0]
    dataset, model
    plt.title(f"{dataset} - {model}")


def plot_error_compression_strategy(df, strat, i):
    strat_df = df[df['strategy'] == strat].sort_values('pruning', ascending=False)
    mean = strat_df.groupby('pruning', as_index=False).mean()
    # std = strat_df.groupby('pruning', as_index=False).std()
    desired_compression = 1/mean['pruning']
    relative_error = (mean['compression']-desired_compression)/desired_compression
    plt.plot(desired_compression, relative_error*100, ls='-', marker='o',
             color=CMAP.colors[i], label=f"{strat[:4]}")


def plot_error_compression(df):
    plt.figure(figsize=(15, 8))
    assert len(set(df['dataset'])) == 1, "More than one dataset in datataframe"
    assert len(set(df['model'])) == 1, "More than one model in datataframe"

    for i, s in enumerate(STRATEGIES):
        plot_error_compression_strategy(df, s, i)

    plt.xscale('log')
    ticks = sorted(set(1/df['pruning']))
    plt.xticks(ticks, map(str,ticks))
    plt.ylabel('% Error')
    plt.xlabel('Desired Compression')
    plt.legend()
    dataset, model, *_ = df.iloc[0]
    dataset, model
    plt.title(f"{dataset} - {model}")
