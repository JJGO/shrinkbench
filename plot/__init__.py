import json
import pathlib
import string

import pandas as pd
import matplotlib.pyplot as plt

from .utils import AutoMap
from .data import df_from_results, df_filter

CMAP = plt.get_cmap('Set1')


colors = AutoMap(CMAP.colors)


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


def plot_error_compression_strategy(df, strat, i, prefix="", ls='-', alpha=1.0):
    strat_df = df[df['strategy'] == strat].sort_values('compression', ascending=False)
    strat_df = strat_df.drop(columns=['completed'])
    mean = strat_df.groupby('compression', as_index=False).mean()
    # std = strat_df.groupby('compression', as_index=False).std()
    desired_compression = mean['compression']
    strat_nick = "".join(c for c in strat if c in string.ascii_uppercase)  # strat[:4]
    relative_error = (mean['real_compression']-desired_compression)/desired_compression
    plt.plot(desired_compression, relative_error*100, ls=ls, marker='o',
             color=CMAP.colors[i], label=f"{prefix}{strat_nick}", alpha=alpha)


def plot_error_compression(df, fig=True, prefix="", ls='-', alpha=1.0):
    if fig:
        plt.figure(figsize=(15, 8))
    assert len(set(df['dataset'])) == 1, "More than one dataset in datataframe"
    assert len(set(df['model'])) == 1, "More than one model in datataframe"

    strategies = [s for s in set(df['strategy']) if s is not None]

    for i, s in enumerate(strategies):
        plot_error_compression_strategy(df, s, i, prefix, ls, alpha)

    plt.xscale('log')
    ticks = [c for c in sorted(set(df['compression'])) if c > 1]
    plt.xticks(ticks, map(str, ticks))
    plt.ylabel('Relative Error [%]')
    plt.xlabel('Desired Compression')
    plt.legend()
    dataset, model, *_ = df.iloc[0]
    dataset, model
    plt.title(f"Relative Compression Error - {dataset} + {model}")
