from collections import defaultdict
from .utils import AutoMap
import matplotlib.pyplot as plt

COLORS = defaultdict(lambda: AutoMap(plt.get_cmap('Set1').colors))
LINES = defaultdict(lambda: AutoMap(['-', '--', ':', '-.']))
MARKERS = defaultdict(lambda: AutoMap(['.', 's', 'v', '^', '<', '>', 'P']))
# LINESTYLES = defaultdict(lambda: AutoMap(product(['-', '--', ':'], ['.', 's', '*'])))


def reset_plt():
    global COLORS, LINES, MARKERS
    COLORS = defaultdict(lambda: AutoMap(plt.get_cmap('Set1').colors))
    LINES = defaultdict(lambda: AutoMap(['-', '--', ':', '-.']))
    MARKERS = defaultdict(lambda: AutoMap(['.', 's', 'v', '^', '<', '>', 'P']))


def plot_df(df,
            x_column,
            y_column,
            colors=None,
            lines=None,
            markers=None,
            delta=False,
            fig=True,
            aggregate=True, **plt_kwargs):
    global COLORS, LINES, MARKERS, LINESTYLES

    colors_column = colors
    lines_column = lines
    markers_column = markers

    if fig:
        plt.figure(figsize=(15, 8))

    groups = []
    if colors_column is not None:
        groups.append(colors_column)
    if lines_column is not None:
        groups.append(lines_column)
    if markers_column is not None:
        groups.append(markers_column)

    labels = set()

    for items, dfg in df.sort_values(by=groups).groupby(groups):
        if not isinstance(items, tuple):
            items = (items, )

        kwargs = {}
        kwargs.update(plt_kwargs)
        label = ""
        if colors_column is not None:
            i, *items = items
            label += f"{i} - "
            kwargs['color'] = COLORS[colors_column][i]

        if lines_column is not None:
            i, *items = items
            label += f"{i} - "
            kwargs['ls'] = LINES[lines_column][i]

        if markers_column is not None:
            i, *items = items
            label += f"{i} - "
            kwargs['marker'] = MARKERS[markers_column][i]

        label = label[:-3]

        dfg = dfg.sort_values(x_column, ascending=True)
        if aggregate:

            kwargs['label'] = label

            dfg = dfg[[x_column, y_column]]
            mean = dfg.groupby(x_column, as_index=False).mean()
            std = dfg.groupby(x_column, as_index=False).std()

            x, y = mean[x_column].values, mean[y_column].values
            yerr = std[y_column]
            if delta:
                y = -(y - y[0])
            plt.errorbar(x, y,  yerr=yerr, **kwargs)
        else:
            for _, dfg_ in dfg.groupby('seed'):

                if label not in labels:
                    labels.add(label)
                    kwargs['label'] = label
                else:
                    kwargs['label'] = None
                dfg_ = dfg_.sort_values(x_column)
                plt.plot(dfg_[x_column].values, dfg_[y_column].values, **kwargs)

    plt.legend()
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    if x_column == 'compression':
        plt.xscale('log')
    ticks = sorted(set(df[x_column]))
    plt.xticks(ticks, map(str, ticks))
    plt.legend()
