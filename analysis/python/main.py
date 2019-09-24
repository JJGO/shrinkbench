#!/usr/bin/env python

from __future__ import print_function, division, absolute_import

import numpy as np
import os
import pandas as pd
import pathlib as pl
import pprint
import matplotlib.pyplot as plt
import seaborn as sb
import time
from matplotlib.ticker import MaxNLocator

import bibtexparser as bib  # pip install bibtexparser
import networkx as nx  # pip install networkx
import scholarly as sc  # pip install scholarly

from python import pareto as par

from joblib import Memory
_memory = Memory('.')

# FIGSIZE_2x1 = (5.5, 7.5)
FIGSIZE_2x1 = (5.5, 5.5)
# PUBLISHED_VS_UNPUB_LABELS = ['Published in Top Venue', 'Other']
# PUBLISHED_VS_UNPUB_LABELS = ['Published', 'Other']
PUBLISHED_VS_UNPUB_LABELS = ['Peer-Reviewed', 'Other']

PAPERS_CSV_PATH = 'assets/existing-results/Papers-Grid view.csv'
RESULTS_CSV_PATH = 'assets/existing-results/Results-Grid view.csv'
MODELS_CSV_PATH = 'assets/existing-results/Models-Grid view.csv'
STATE_OF_SPARSITY_DIR = 'assets/existing-results/state-of-sparsity-resnet50'
LOTTERY_TICKET_DIR = 'assets/existing-results/lottery-tix'
EFFICIENTNET_RESULTS_DIR = 'assets/existing-results/efficientnet'
KERAS_MODEL_INFO_DIR = 'assets/existing-results/keras'
PYTORCH_PRETRAINED_INFO_DIR = 'assets/existing-results/pytorch-pretrained'


ALL_VALID_DSET_MODEL_COMBOS = [
    ('ImageNet', 'VGG-16'),
    ('ImageNet', 'AlexNet'), ('ImageNet', 'CaffeNet'), ('ImageNet', 'Alex/CaffeNet'),
    ('ImageNet', 'ResNet-18'), ('ImageNet', 'ResNet-34'),
    ('ImageNet', 'ResNet-50'),
    ('CIFAR-10', 'ResNet-56')]

VALID_DSET_MODEL_COMBOS_COMBINE_ALEXLIKE = ALL_VALID_DSET_MODEL_COMBOS
VALID_DSET_MODEL_COMBOS_COMBINE_ALEXLIKE.remove(('ImageNet', 'AlexNet'))
VALID_DSET_MODEL_COMBOS_COMBINE_ALEXLIKE.remove(('ImageNet', 'CaffeNet'))

OUTPUT_SAVE_DIR = pl.Path('results/survey')
FIGS_SAVE_DIR = pl.Path('figs/survey')

METHOD_RANDOM = 'Gale 2019, Random'

ALL_NOT_HIDEOUS_MARKERS = ['o', 'v', '^', '<', '>', '1', '2',
                           'h', 's', 'p', '*', 'x', 'X', 'd', 'D',
                           r'$\int$', r'$\cup$', r'$\cap$', r'$\bigwedge$',
                           r'$\bigvee$']
ALL_NOT_HIDEOUS_LINE_STYLES = ['-', '--', '-.', ':']

DEFAULT_SEABORN_STYLE = "darkgrid"
sb.set_style(DEFAULT_SEABORN_STYLE)
DEFAULT_CMAP = plt.get_cmap('tab10')
DEFAULT_CMAP_NCOLORS = 10


def color_for_idx(idx, cmap=None, cmap_ncolors=10):
    if cmap is None:
        cmap = cmap or DEFAULT_CMAP
        cmap_ncolors = DEFAULT_CMAP_NCOLORS
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    return cmap((idx % cmap_ncolors) / cmap_ncolors + .01)


def marker_for_idx(idx, markerlist=None):
    if markerlist is None:
        markerlist = ALL_NOT_HIDEOUS_MARKERS
    return markerlist[idx % len(markerlist)]


def hist_integers(ax, ints, allow_zero=True, include_cdf=False, **hist_kwargs):
    minval = -.5 if allow_zero else .5

    # maxval = np.max(np.asarray(ints).ravel())
    maxval = np.max([np.max(ints) for ints in ints])
    # print("maxval: ", maxval)
    # print("largest ints: ", np.sort(ints)[::-1][:10])

    # print("ints as array: ", np.asarray(ints).ravel())
    # print("minval: ", minval)
    # print("maxval: ", maxval)

    bins = np.arange(minval, maxval + 1)  # unclear why +1 is needed
    # ax.hist(ints, bins, cumulative=True)
    axes = [ax]

    ax.hist(ints, bins, **hist_kwargs)

    if include_cdf:
        # plot cdf
        ax2 = ax.twinx()
        color = list(color_for_idx(1))
        color[-1] = .25  # set opacity to less than 1
        # print("color:", color)
        # ax.hist(ints, bins, cumulative=True, color=color)
        ax2.hist(ints, bins, cumulative=True, color=color, zorder=0, density=True)

        # see http://matplotlib.1069221.n5.nabble.com/Control-twinx-series-zorder-ax2-series-behind-ax1-series-or-place-ax2-on-left-ax1-on-right-td12994.html # noqa
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)  # magic to make new z-order work

    # else:
    #     ax.hist(ints, bins)  # just plot pdf, and no need to mess with style

        # ax2.hist(ints, bins)
        # ax, ax2 = ax2, ax
        # axes.append(ax2)
    # else:
    #     ax.hist(ints, bins)

    for ax in axes:
        # sb.despine(top=True, right=False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim([minval, maxval + .5])

    return ax, ax2 if include_cdf else ax


def lotsa_colors_cmap(value):
    assert 0 <= value <= 1  # if this throws, I don't understand cmaps
    if value < .3333:
        return plt.get_cmap('tab20')(3 * value)
    elif value < .6666:
        return plt.get_cmap('tab20b')((3 * value) - 1)
    else:
        return plt.get_cmap('tab20c')((3 * value) - 2)


def make_line_traits_dicts(df, color_colname=None, marker_colname=None,
                           line_style_colname=None):
    val2color = None
    val2marker = None
    val2linestyle = None
    if color_colname:
        uniq_vals = df[color_colname].unique()
        # scale_by = len(all_vals)
        # colors = [lotsa_colors_cmap(i / scale_by)
        need_ncolors = len(uniq_vals)
        if need_ncolors <= 10:
            # cmap = plt.get_cmap('gist_earth')
            cmap = plt.get_cmap('tab10')
            # colors = [cmap(positions[i]) for i in range(len(uniq_vals))]
            # positions = [9, 0, 1, 3, 2, 4, 5, 7, 6, 8]
            positions = [8, 1, 3, 4, 0, 9, 2, 7, 5, 6]
            colors = [cmap(positions[i] / 9.5) for i in range(len(uniq_vals))]
            # cmap = plt.get_cmap('plasma')
            # positions = [.98, .84, .7, .56, .42, .28, .05, .01]
            # colors = [cmap(positions[i]) for i in range(len(uniq_vals))]
            # cmap = plt.get_cmap('CMRmap')
            # colors = [cmap((i + 2 % 10) / 10 + .01) for i in range(len(uniq_vals))]
        else:
            colors = [lotsa_colors_cmap((i % 60) / 60 + .01)  # has 60 colors
                      for i in range(len(uniq_vals))]
        val2color = dict(zip(uniq_vals, colors))
    if marker_colname:
        uniq_vals = df[marker_colname].unique()
        markers = [ALL_NOT_HIDEOUS_MARKERS[i % len(ALL_NOT_HIDEOUS_MARKERS)] for
                   i in range(len(uniq_vals))]
        val2marker = dict(zip(uniq_vals, markers))
    if line_style_colname:
        uniq_vals = df[line_style_colname].unique()
        line_styles = [
            ALL_NOT_HIDEOUS_LINE_STYLES[i % len(ALL_NOT_HIDEOUS_LINE_STYLES)]
            for i in range(len(ALL_NOT_HIDEOUS_LINE_STYLES))]
        val2linestyle = dict(zip(uniq_vals, line_styles))

    return val2color, val2marker, val2linestyle


if not os.path.exists(OUTPUT_SAVE_DIR):
    OUTPUT_SAVE_DIR.mkdir(parents=True)

if not os.path.exists(FIGS_SAVE_DIR):
    FIGS_SAVE_DIR.mkdir(parents=True)


def save_fig(name):
    plt.savefig(os.path.join(FIGS_SAVE_DIR, name + '.png'),
                dpi=300, bbox_inches='tight')


@_memory.cache
def _cached_search_pubs_query_top_hit(query_str):
    time.sleep(1 + np.abs(np.random.randn() * 2))
    return next(sc.search_pubs_query(query_str))


def info_for_bibtex(bibtex_str, cached=True, include_scholar_info=True):
    # airtable download sometimes preprends a "'" for no reason
    entries = bib.loads(bibtex_str.strip('').strip("'")).entries

    if not include_scholar_info:
        return entries

    if len(entries) < 1:
        print("ERROR: found no bibtex entries in bibtex:")
        print(bibtex_str)
        print("entries object: ", entries)

    combined_infos = []
    # for citekey in info.entries_dict:
    for entry in entries:

        # paper_info_dict = info.entries_dict[citekey]
        # query_str = '"{}"'.format(paper_info_dict['title'])  # need exact match
        # query_str = '{}'.format(paper_info_dict['title'])
        # query_str = '{}'.format(entry['title'])
        query_str = '"{}"'.format(entry['title'])
        if cached:
            try:
                sc_info = _cached_search_pubs_query_top_hit(query_str)
            except StopIteration as e:
                print("Google Scholar is angry at us. Try waiting a while")
                raise(e)
        else:
            print("searching google scholar for query_str: ", query_str)
            sc_info = next(sc.search_pubs_query(query_str))  # just take top hit

        # print("got sc_info: ", sc_info)
        # combined_infos.append((paper_info_dict, sc_info))
        combined_infos.append((entry, sc_info))

    if len(combined_infos) < 1:
        print("ERROR: found no bibtex entries in bibtex:")
        print(bibtex_str)

    return combined_infos if len(combined_infos) > 1 else combined_infos[0]


def load_clean_papers_csv(load_scholar_info=False, drop_bibtex=True):
    # return pd.read_csv(PAPERS_CSV_PATH)
    df = pd.read_csv(PAPERS_CSV_PATH)

    # drop unused cols
    for col in ['Notes', 'Compares to names', 'Results',
                'Datasets 2', 'Models 3']:
        try:
            df.drop([col], axis=1, inplace=True)
        except KeyError:
            pass
    try:
        df.drop('Attachments', axis=1, inplace=True)
    except KeyError:
        pass  # one of these days I'll delete this col, so future-proof code

    # create pretty name for paper
    # df['Year'] = [s.split('-')[0] for s in df['Date']]
    # mask = ~df['Year'].isna()
    # df['Year'].loc[mask] = df['Date'].loc[mask].apply(lambda s: s.split('-')[0])
    df['Year'] = df['Date'].apply(lambda s: s.split('-')[0])
    df['PrettyName'] = ['{} {}'.format(name, year)
                        for name, year in zip(df['AuthorName'], df['Year'])]

    # make it possible to join with results
    df.rename({'Name': 'Paper'}, axis=1, inplace=True)

    # print("papers df dtypes:")
    # print(df.dtypes)
    # print(df)

    # df = df.head(5)
    # print(df[['Name', 'PrettyName', 'AuthorName', 'Year']])


    # print("load_clean_papers_csv():")
    # row = df.iloc[1]
    # print(row)
    # print(row['BibTeX'])

    # sc_info = google_scholar_info_for_bibtex(row['BibTeX'])

    # print(sc_info)
    # pprint.pprint(sc_info.__dict__)
    # print(sc_info.citedby)
    # print(sc_info.bib['title'])

    # return

    # df = df.head(5)

    # nrows = df.shape[0]
    # df = df.iloc[::-1]

    if not load_scholar_info:
        if drop_bibtex:
            df.drop('BibTeX', axis=1, inplace=True)
        return df

    bibtex_strs = list(df['BibTeX'])
    ncitations = []
    titles = []
    for s in bibtex_strs:
        bibtex_dict, sc_info = info_for_bibtex(s)
        # pprint.pprint(sc_info.__dict__)

        title = bibtex_dict['title']
        try:
            ncites = sc_info.citedby
        except AttributeError:
            ncites = 0  # scholar returns no citedby if not cited
        if 'state of sparsity in deep neural networks' in title.lower():
            ncites = 12   # kludge since google scholar can't find this paper

        titles.append(title)
        ncitations.append(ncites)
        print('"{}": {} cites'.format(title, ncites))

    df['Citations'] = ncitations
    df['Title'] = titles

    # print(df)

    if drop_bibtex:
        df.drop('BibTeX', axis=1, inplace=True)
    return df


def make_papers_comparisons_fig(save=True, plot_dep_graph=False):
    # df = load_papers_csv()
    df = load_clean_papers_csv()

    # print(df.shape)
    df = df[~df['Date'].isna()]
    # print(df.shape)
    # print(df)
    # return

    print(df.shape)
    print(len(df.columns))
    print(df.columns)

    paper_ids = df['Paper']
    # authors = df['author']
    compares_to_ids = df['Compares To']
    dates = list(df['Date'])

    valid_ids = set(paper_ids)
    # assert len(paper_ids) == len(dates)

    G = nx.DiGraph()

    # print(compares_to_ids)

    for pid in paper_ids:
        G.add_node(pid)

    for paper_id, other_ids in zip(paper_ids, compares_to_ids):
        print(other_ids)
        if isinstance(other_ids, float):  # nan
            continue
        for other_id in other_ids.split(','):
            other_id = other_id.strip()
            print("{} -> {}".format(paper_id, other_id))

            if other_id in valid_ids:
                G.add_edge(paper_id, other_id)

    # ------------------------ print/plot number of papers that compare to it

    # _, axes = plt.subplots(1, 2, figsize=(11, 4))
    _, axes = plt.subplots(2, 1, figsize=FIGSIZE_2x1)

    # indegrees = G.in_degree(with_labels=True)

    title_fontsize = 14
    fontsize = 12.2

    indegrees = [(pid, G.in_degree(pid)) for pid in paper_ids]
    pprint.pprint(sorted(indegrees, key=lambda tup: tup[1], reverse=True))
    degrees_ar = np.array([deg[1] for deg in indegrees])
    ax = axes[0]
    ax.set_title('Number of Papers Comparing to a Given Paper',
                 fontsize=title_fontsize)
    ax.set_xlabel('Compared to by this many other papers', fontsize=fontsize)
    ax.set_ylabel('Number of papers\ncompared to this many times',
                  fontsize=fontsize)

    id2published = _load_is_published_at_top_conf_dict()
    is_pub = [id2published[pid] for pid in paper_ids]

    split_by_venue = True
    stacked = True

    colors = [color_for_idx(1), color_for_idx(0)]

    if split_by_venue:
        degrees_published = [deg for i, deg in enumerate(degrees_ar)
                             if is_pub[i]]
        degrees_unpublished = [deg for i, deg in enumerate(degrees_ar)
                               if not is_pub[i]]
        # hist_integers(ints=degrees_unpublished, ax=ax)
        # hist_integers(ints=degrees_published, ax=ax, bottom=degrees_unpublished)
        hist_integers(ints=[degrees_published, degrees_unpublished],
                      ax=ax, stacked=stacked, color=colors)
    else:
        hist_integers(ints=degrees_ar, ax=ax)

    # ax.set_xlim([0, ax.get_xlim()[1]])

    # ------------------------ print/plot number of papers they compare to

    outdegrees = [(pid, G.out_degree(pid)) for pid in paper_ids]
    outdegrees = sorted(outdegrees, key=lambda tup: tup[1], reverse=True)

    pprint.pprint(outdegrees)

    degrees_ar = np.array([deg[1] for deg in outdegrees])

    ax = axes[1]
    ax.set_title('Number of Papers a Given Paper Compares To',
                 fontsize=title_fontsize)
    ax.set_xlabel('Compares to this many other papers', fontsize=fontsize)
    ax.set_ylabel('Number of papers that\ncompare to this many others',
                  fontsize=fontsize)

    if split_by_venue:
        degrees_published = [deg for i, deg in enumerate(degrees_ar)
                             if is_pub[i]]
        degrees_unpublished = [deg for i, deg in enumerate(degrees_ar)
                               if not is_pub[i]]
        hist_integers(ints=[degrees_published, degrees_unpublished],
                      ax=ax, stacked=stacked, label=PUBLISHED_VS_UNPUB_LABELS,
                      color=colors)
    else:
        hist_integers(ints=degrees_ar, ax=ax)
    # ax.hist(degrees_ar, bins=np.max(degrees_ar)+1)
    # ax.set_xlim([0, ax.get_xlim()[1]])
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()

    if split_by_venue:
        handles, labels = ax.get_legend_handles_labels()
        plt.figlegend(handles, labels, loc='lower center', ncol=2)
        # plt.subplots_adjust(bottom=.11)
        plt.subplots_adjust(bottom=.15)

    if save:
        save_fig('paper_comparisons_hist')
    else:
        plt.show()

    # ------------------------ print/plot dependency graph

    if not plot_dep_graph:
        return

    dates_years = np.array([int(date.split('-')[0]) for date in dates])
    dates_months = np.array([int(date.split('-')[1]) for date in dates])

    x_positions = dates_months + (dates_years * 12)
    x_positions = np.maximum(0, x_positions - (2013 * 12))

    # print(x_positions)
    # print(sorted(paper_ids))
    # return

    # y_positions = np.random.randint(10, size=len(x_positions))
    y_positions = x_positions % 12
    # y_positions = (x_positions % 12)**1.5
    # y_positions = np.sqrt(x_positions % 12)

    pos = {pid: (x_positions[i], y_positions[i])
           for i, pid in enumerate(paper_ids)}  # noqa

    # TODO use (node_color, vmin, vmax, cmap) and
    # (edge_color, edge_vmin, edge_vmax, edge_cmap) to color code so that it's
    # obvious what's comparing to what and so that it doesn't look more
    # connected than it is from edges that go behind everything within one year

    print(G.nodes)
    print(G.edges)
    nx.draw(G, with_labels=False, pos=pos)

    plt.show()


def load_lottery_ticket_resnet_results():
    p = pl.Path(LOTTERY_TICKET_DIR)

    # reset to epoch dir
    df = pd.read_csv(p / 'fig5-resnet50-reset-to-epoch-vs-top1.csv')

    # PaperName = 'lottery-ticket-followup'  # CamelCase like key in case refactor
    # base_dict = {'Dataset': 'ImageNet',
    base_dict = {'PaperName': 'lottery-ticket-followup',
                 'Dataset': 'ImageNet',
                 'Model': 'ResNet-50',
                 'OrigTop1': 76.14}  # +/- .08

    acc_fmt_str = 'fracRemaining={}%'
    std_fmt_str = 'stdFrac{}%'  # technically max deviation, not std

    # ------------------------ results from resetting to different epochs
    results = []
    for _, row in df.iterrows():
        base_row_dict = base_dict.copy()
        row = dict(row)

        epochnum = row['resetToEpoch'].strip()
        base_row_dict['Method'] = 'Frankle 2019, ResetToEpoch={}'.format(epochnum)
        row.pop('resetToEpoch')

        for percent in [10, 20, 30, 50, 70]:
            acc_col = acc_fmt_str.format(percent)
            std_col = std_fmt_str.format(percent)
            result = base_row_dict.copy()
            result['Top1'] = row[acc_col]
            result['stdTop1'] = row[std_col]
            # result['keptParams'] = kept_params_fmt_str.format(percent)
            result['keptParams'] = percent
            results.append(result)

    df_reset_when = pd.DataFrame.from_records(results)

    # ------------------------ results from pruning at different epochs
    df = pd.read_csv(p / 'fig9-resnet50-turnaround-epoch-vs-top1.csv')
    results = []
    for _, row in df.iterrows():
        base_row_dict = base_dict.copy()
        row = dict(row)

        epochnum = row['pruneAtEpoch']
        base_row_dict['Method'] = 'Frankle 2019, PruneAtEpoch={}'.format(int(epochnum))
        row.pop('pruneAtEpoch')

        for percent in [10, 20, 30, 50, 70]:
            acc_col = acc_fmt_str.format(percent)
            std_col = std_fmt_str.format(percent)
            result = base_row_dict.copy()
            result['Top1'] = row[acc_col]
            result['stdTop1'] = row[std_col]
            # result['keptParams'] = kept_params_fmt_str.format(percent)
            result['keptParams'] = percent
            results.append(result)

    df_prune_when = pd.DataFrame.from_records(results)

    df = pd.concat((df_reset_when, df_prune_when), axis=0, ignore_index=True)
    df.sort_values(['Method', 'keptParams'], inplace=True)

    df['OrigTop1'] /= 100.
    df['Top1'] /= 100.
    df['stdTop1'] /= 100.
    df['keptParams'] /= 100.

    return df


def load_efficientnet_results():
    df = pd.read_csv(
        pl.Path(EFFICIENTNET_RESULTS_DIR) / 'efficientnet-models-summary.csv')
    df['Dataset'] = 'ImageNet'
    # NOTE: have to have no leading space in csv file for these keys
    # or pandas adds it as part of the key for no clear reason
    df.drop('#Params-Ratio-to-EfficientNet', inplace=True, axis=1)
    df.drop('#FLOPS-Ratio-to-EfficientNet', inplace=True, axis=1)

    methods, years = _methods_and_years_for_models(df['Model'])

    df['Method'] = methods
    df['Year'] = years

    df.sort_values('Year', axis=0, inplace=True)

    df['Top1'] /= 100.
    df['Top5'] /= 100.

    return df


def _methods_and_years_for_models(models):
    models = list(models)
    methods = []
    years = []
    for orig_model_name in models:
        model = orig_model_name.lower()
        if 'efficient' in model:
            tup = ('EfficientNet (2019)', 2019)
        elif 'amoeba' in model:
            tup = ('AmoebaNet (2018/2019)', 2018.5)
        elif model == 'bn-inception':  # has to be before *ception
            tup = ('BN-Inception (2015)', 2015)
        elif 'ception' in model:
            tup = ('{In,X}ception (2016/2017)', 2016.5)
        elif model.startswith('se'):  # has to be before resnet case
            tup = ('SENet (2018)', 2018)
        elif 'resnet' in model:
            tup = ('ResNet (2015/2016)', 2015.5)
        elif 'resnext' in model:
            tup = ('ResNeXt (2017)', 2017)
        elif 'densenet' in model:
            tup = ('DenseNet (2017)', 2017)
        elif 'nasnet' in model:
            tup = ('NASNet (2018)', 2018)
        elif model == 'polynet':
            tup = ('Polynet (2017)', 2017)
        elif model == 'gpipe':
            tup = ('GPipe (2018)', 2018)
        elif model == 'googlenet':
            tup = ('GoogLeNet (2015)', 2015)
        elif model.endswith('_bn'):
            tup = ('VGG+Batchnorm', 2015)
        elif model.startswith('vgg'):
            tup = ('VGG (2014)', 2014)
        elif model.startswith('dpn'):
            tup = ('DPN (2017)', 2017)
        elif model == 'mobilenet-v1':
            tup = ('MobileNet-v1 (2017)', 2017)
        elif model == 'mobilenet-v2':
            tup = ('MobileNet-v2 (2018)', 2018)
        elif model.startswith('squeezenet'):
            tup = ('SqueezeNet (2016)', 2016)
        elif model.startswith('shufflenet'):
            tup = ('ShuffleNet (2017)', 2017)
        elif model == 'alexnet':
            tup = ('AlexNet (2012)', 2012)
        else:
            raise ValueError('unrecognized model: {}'.format(model))
        methods.append(tup[0])
        years.append(tup[1])

    return methods, years


def load_pytorch_pretrained_models_csv():
    df = pd.read_csv(
        pl.Path(PYTORCH_PRETRAINED_INFO_DIR) / 'ModelsComparison.csv')

    df.rename({'FLOPs': 'GFLOPS',
               'Top-1 (%)': 'Top1',
               'Top-5 (%)': 'Top5'}, axis=1, inplace=True)
    df['MParams'] = df['# Parameters'].values / 1e6
    df['Dataset'] = 'ImageNet'
    df.drop(['# Parameters', 'Parameters (MB)'], axis=1, inplace=True)

    methods, years = _methods_and_years_for_models(df['Model'])
    df['Method'] = methods
    df['Year'] = years

    df.sort_values('Year', axis=0, inplace=True)

    df['Top1'] /= 100.
    df['Top5'] /= 100.

    return df


def load_existing_models_results():
    df = load_efficientnet_results()
    df2 = load_pytorch_pretrained_models_csv()
    df = pd.concat([df, df2], axis=0, ignore_index=True, sort=False)
    df.sort_values('Year', axis=0, inplace=True)
    return df


def load_models_csv():
    df = pd.read_csv(MODELS_CSV_PATH)
    df.drop(['Notes', 'Source', 'Results', 'Count', 'Number of Papers'],
            inplace=True, axis=1)
    return df


def load_state_of_sparsity_resnet_results():
    p = pl.Path(STATE_OF_SPARSITY_DIR)
    # csv_paths = list(p.glob('*.csv'))
    # print(csv_paths)

    # clean up baselines csv
    df = pd.read_csv(p / 'rn50_baselines.csv')
    df.rename({'Accuracy': 'Top1'}, axis=1, inplace=True)
    df['Dataset'] = 'ImageNet'
    df['Model'] = 'ResNet-50'
    print("df.columns", df.columns)
    # df_baseline = df[['Dataset', 'Model', 'Top1']]
    baseline_acc_mean = df['Top1'].mean() / 100
    assert .7 < baseline_acc_mean < .8
    # baseline_acc_std = df['Top1'].std() / 100  # sample std (N-1)

    # baseline_acc_str = '{}%%'.format(baseline_acc_mean)

    # clean up hard concrete distro results
    df = pd.read_csv(p / 'rn50_l0_regularization.csv')
    df.rename({'accuracy': 'Top1'}, axis=1, inplace=True)
    df.rename({'global_sparsity': 'savedParams'}, axis=1, inplace=True)
    # df['Method'] = 'HardConcrete'
    df['Method'] = 'Gale 2019, HardConcrete'
    df['Date'] = '2017-12'
    df['OrigTop1'] = baseline_acc_mean
    df['Dataset'] = 'ImageNet'
    df['Model'] = 'ResNet-50'
    # df['Top1'] *= 100  # for this csv only, numbers aren't percentages
    df_hard_concrete = df

    # clean up magnitude pruning results
    df = pd.read_csv(p / 'rn50_magnitude_pruning.csv')
    df.rename({'accuracy': 'Top1'}, axis=1, inplace=True)
    df.rename({'Sparsity': 'savedParams'}, axis=1, inplace=True)
    df['Method'] = 'Gale 2019, Magnitude'
    # df['OrigTop1'] = baseline_acc_str
    df['Date'] = '2015-12'
    df['OrigTop1'] = baseline_acc_mean
    df['Dataset'] = 'ImageNet'
    df['Model'] = 'ResNet-50'
    df_magnitude = df

    # clean up random pruning results
    df = pd.read_csv(p / 'rn50_random_pruning.csv')
    df.rename({'accuracy': 'Top1'}, axis=1, inplace=True)
    df.rename({'Target Sparsity': 'savedParams'}, axis=1, inplace=True)
    # df['Paper'] = METHOD_RANDOM  # not an actual paper
    df['Date'] = 'N/A'
    # df['Method'] = 'Gale 2019, ' + METHOD_RANDOM  # not an actual paper
    df['Method'] = METHOD_RANDOM  # not an actual paper
    df['OrigTop1'] = baseline_acc_mean
    df['Dataset'] = 'ImageNet'
    df['Model'] = 'ResNet-50'
    df_random = df

    # clean up random pruning results
    df = pd.read_csv(p / 'rn50_variational_dropout.csv')
    df.rename({'Accuracy': 'Top1'}, axis=1, inplace=True)
    df.rename({'global sparsity': 'savedParams'}, axis=1, inplace=True)
    # df['Paper'] = 'sparse-variational-dropout'
    df['Method'] = 'Gale 2019, SparseVD'
    df['Date'] = '2017-6'
    df['OrigTop1'] = baseline_acc_mean
    df['Dataset'] = 'ImageNet'
    df['Model'] = 'ResNet-50'
    df_vd = df

    df_combined = pd.concat(
        [df_hard_concrete, df_magnitude, df_random, df_vd],
        # axis=0, ignore_index=True, join='inner')
        axis=0, ignore_index=True, join='outer')
    # df_combined = df_combined[['Paper', 'Dataset', 'Model', 'Top1',
    df_combined = df_combined[['Method', 'Dataset', 'Model', 'Top1',
                               'savedParams', 'OrigTop1', 'Date']]
    df_combined['Year'] = df_combined['Date'].apply(lambda s: s.split('-')[0])

    return df_combined


# @_memory.cache
def load_pruning_results():
    df = pd.read_csv(RESULTS_CSV_PATH)
    df.drop('Name', axis=1, inplace=True)  # this col doesn't mean anything
    df.drop('Notes', axis=1, inplace=True)  # unused

    # ------------------------ rm / combine problematic methods

    # these results are not really comparable to other results I tabulated
    df = df[df['Paper'] != 'lempitsky-fast-convnets']

    # spectral-pruning GAP results change the architecture; Spec-Tiny performs
    # terribly and messes up the ylim on the plots; Spec-Conv2 appears to
    # just be a variation that provides most direct comparison to Thinet
    # (and also has exactly same nparams as another one, and therefore messes
    # up the plots)
    df = df[~df['Method'].isin(('Spec-GAP', 'Spec-GAP2', 'Spec-GAPe',
                                'Spec-Tiny', 'Spec-Conv2'))]

    # iffy whether the remaining variations of their approach should be
    # combined or not; going ahead and combining them since they seem to
    # be at different operating points and it makes the plots cleaner
    df['Method'].loc[df['Paper'] == 'spectral-pruning'] = np.nan

    # ------------------------ generic cleaning

    if 'stdTop5' not in df.columns:
        df['stdTop5'] = np.NaN  # nothing reports this yet
    PERCENT_KEYS = ('OrigTop1 OrigTop5 OrigErr1 OrigErr5 Err1 Err5 Top1 Top5 '
                    'dTop1 dTop5 savedFLOPS savedParams keptFLOPS keptParams '
                    'stdTop1 stdTop5').split()

    for col in PERCENT_KEYS:
        # df[col] = df[col].str.rstrip('%').astype('float') / 100.
        # print("col: ", col)
        df[col] = df[col].apply(
            lambda s: (float(s.rstrip('%'))) / 100.
            if isinstance(s, str) else s).astype('float')
        assert df[col].max(skipna=True) < 1. or (col == 'stdTop5')
        if col not in ('dTop1', 'dTop5'):
            # print(df[col].min(skipna=True))
            df[col].min(skipna=True) >= 0
        assert df[col].isna().sum() > 0  # every percent col has missing data

    # add in relevant paper info (this should be joins but pandas hates me)
    papers_df = load_clean_papers_csv()

    df = df.join(papers_df.set_index('Paper'), on='Paper', how='left')
    df.drop(['Models', 'Datasets'], axis=1, inplace=True)

    # ------------------------ create pretty method names

    mask = df['Method'].isna()
    df['Method'].loc[mask] = df['PrettyName'].loc[mask]
    existing_methods = df['Method'].loc[~mask]
    their_paper_names = df['PrettyName'].loc[~mask]
    new_method_names = ['{}, {}'.format(name, method)
                        for method, name in zip(
                            existing_methods, their_paper_names)]
    df['Method'].loc[~mask] = new_method_names

    return df


def clean_results(df, rm_reimplementations=True):
    if rm_reimplementations:
        df = df.loc[~df['Method'].isin(
            ['Magnitude', 'Gale 2019, Magnitude', 'Gale 2019, SparseVD',
             'Gale 2019, HardConcrete', 'Liu 2019, Magnitude'])]

    # ------------------------ impute missing acc fields

    # handle missing cols without messing with logic below
    used_keys = 'Top1 Top5 Err1 Err5'.split()
    used_keys += 'OrigTop1 OrigTop5 OrigErr1 OrigErr5'.split()
    used_keys += 'dTop1 dTop5 MParams OrigMParams GFLOPS OrigGFLOPS'.split()
    used_keys += 'keptParams keptFLOPS savedParams savedFLOPS'.split()
    used_keys += 'stdTop1 stdTop5 NumTrials'.split()
    used_keys += ['old/new Nparam', 'old/new FLOPS']
    for k in used_keys:
        if k not in df.columns:
            df[k] = np.NaN

    # fill in top1/top5 acc from err rates
    df['Top1'].loc[df['Top1'].isna()] = 1. - df['Err1']
    df['Top5'].loc[df['Top5'].isna()] = 1 - df['Err5']
    # fill in err rates because why not
    try:
        df['Err1'].loc[df['Err1'].isna()] = 1. - df['Top1']
        df['Err5'].loc[df['Err5'].isna()] = 1. - df['Top5']
    except KeyError:
        pass

    # fill in orig top1/top5 acc + err
    df['OrigTop1'].loc[df['OrigTop1'].isna()] = 1. - df['OrigErr1']
    df['OrigTop5'].loc[df['OrigTop5'].isna()] = 1 - df['OrigErr5']
    df['OrigErr1'].loc[df['OrigErr1'].isna()] = 1. - df['OrigTop1']
    df['OrigErr5'].loc[df['OrigErr5'].isna()] = 1. - df['OrigTop5']

    # fill in change in top1/top5 from orig and err rates
    mask = ~(df['OrigTop1'].isna() | df['Top1'].isna())
    df['dTop1'].loc[mask] = (df['Top1'] - df['OrigTop1']).loc[mask]
    mask = ~(df['OrigTop5'].isna() | df['Top5'].isna())
    df['dTop5'].loc[mask] = (df['Top5'] - df['OrigTop5']).loc[mask]

    # fill in orig top1/top5 from top1/5 and dTop1/5
    mask = (~df['Top1'].isna()) & (~df['dTop1'].isna()) & df['OrigTop1'].isna()
    df['OrigTop1'].loc[mask] = (df['Top1'] + df['dTop1']).loc[mask]

    # ------------------------ impute missing size/speed fields

    # infer old/new Nparams, keptParams, savedParams
    mask = (~df['MParams'].isna()) & (~df['OrigMParams'].isna())
    df['keptParams'].loc[mask] = (df['MParams'] / df['OrigMParams']).loc[mask]
    df['keptParams'].loc[df['keptParams'].isna()] = 1. - df['savedParams']
    df['old/new Nparam'].loc[df['old/new Nparam'].isna()] = 1. / df['keptParams']

    df['keptParams'].loc[df['keptParams'].isna()] = 1. / df['old/new Nparam']
    df['savedParams'].loc[df['savedParams'].isna()] = 1. - df['keptParams']

    # infer old/new FLOPS, keptFlops, savedFLOPS
    mask = (~df['GFLOPS'].isna()) & (~df['OrigGFLOPS'].isna())
    df['keptFLOPS'].loc[mask] = (df['GFLOPS'] / df['OrigGFLOPS']).loc[mask]
    df['keptFLOPS'].loc[df['keptFLOPS'].isna()] = 1. - df['savedFLOPS']
    df['old/new FLOPS'].loc[df['old/new FLOPS'].isna()] = 1. / df['keptFLOPS']

    df['keptFLOPS'].loc[df['keptFLOPS'].isna()] = 1. / df['old/new FLOPS']
    df['savedFLOPS'].loc[df['savedFLOPS'].isna()] = 1. - df['keptFLOPS']

    # infer orig nparams and flops
    df['OrigMParams'].loc[df['OrigMParams'].isna()] = \
        df['MParams'] / df['keptParams']
    df['OrigGFLOPS'].loc[df['OrigGFLOPS'].isna()] = \
        df['GFLOPS'] / df['keptFLOPS']

    # ------------------------ misc imputation / cleaning

    df.rename({'old/new Nparam': 'Compression Ratio',
               'old/new FLOPS': 'Theoretical Speedup'}, axis=1, inplace=True)

    # assume only one trial if not otherwise noted
    df['NumTrials'].fillna(1)

    # create "Alex/CaffeNet" model id for union of AlexNet and CaffeNet results
    dfAlex = df[df['Model'] == 'AlexNet']
    dfCaffe = df[df['Model'] == 'CaffeNet']
    dfAlexLike = pd.concat([dfAlex, dfCaffe], axis=0)
    dfAlexLike['Model'] = 'Alex/CaffeNet'
    df = pd.concat([df, dfAlexLike], axis=0)

    return df


@_memory.cache
def load_results(include_pruning=False, include_goog_results=False,
                 include_lottery_ticket=False, include_efficientnet=False,
                 clean=True, **clean_kwargs):
    df = None
    if include_pruning:
        df = load_pruning_results()
    if include_goog_results:
        df_goog = load_state_of_sparsity_resnet_results()
        df = df_goog if df is None else pd.concat(
            (df, df_goog), axis=0, ignore_index=True, sort=False)
    if include_lottery_ticket:
        df_lot = load_lottery_ticket_resnet_results()
        df = df_lot if df is None else pd.concat(
            (df, df_lot), axis=0, ignore_index=True, sort=False)
    if include_efficientnet:
        df_eff = load_efficientnet_results()
        df = df_eff if df is None else pd.concat(
            (df, df_eff), axis=0, ignore_index=True, sort=False)

    if clean:
        df = clean_results(df, **clean_kwargs)

    return df


def estimated_model_stats():
    df = load_results(include_pruning=True)
    df = df[~(df['OrigMParams'].isna() & df['OrigGFLOPS'].isna())]

    all_models = list(df['Model'].unique())
    print('all_models:', all_models)

    keep_cols = ['Model', 'Paper', 'Venue', 'Dataset', 'OrigTop1', 'OrigTop5',
                 'OrigMParams', 'OrigGFLOPS']
    for model in all_models:
        subdf = df[df['Model'] == model][keep_cols]
        print("\n------------------------")
        print(subdf)


def plot_tradeoff_curve(df, dset, model, xcol, ycol, min_acc=.5,
                        save=False, ax=None, semilogx=False,
                        val2color=None, val2marker=None):
    # select only rows with appropriate model and dataset
    df = df.loc[df['Dataset'] == dset]
    if model is not None:
        df = df.loc[df['Model'] == model]

    # rm rows that don't have reported top1 or top5 acc below min_acc (this
    # should only affect the state-of-sparsity results, which get crazy
    # speedups on some trials but do no better than chance)
    # df = df.loc[~(df['Top1'] < min_acc) | (df['Top5'] < min_acc)]
    # df = df.loc[~(df['Top1'] < min_acc) | (df['Top5'] < min_acc)]

    # rm rows with nan vals for xcol or ycol
    df = df.loc[~(df[xcol].isna() | df[ycol].isna())]

    # plt.figure(figsize=(10, 8))
    given_ax = ax is not None
    if not given_ax:
        _, ax = plt.subplots(figsize=(10, 8))

    print('plotting {}_{}_{}_{}'.format(dset, model, xcol, ycol))
    # assert given_ax # TODO rm

    sort_by = [xcol]
    if 'Date' in df.columns:
        sort_by = ['Date'] + sort_by
    df = df.sort_values(sort_by, axis=0)

    for method in df['Method'].unique():
        subdf = df[df['Method'] == method]

        x = subdf[xcol].values
        y = subdf[ycol].values * 100

        # ------------------------ compute std deviation for err bars
        new_x = np.unique(x)
        new_y = np.unique(y)
        new_y = []
        stds = []
        for xval in new_x:
            idxs = np.where(x == xval)[0]
            yvals = y[idxs]

            if len(idxs) > 1:
                print("method {} has multiple ys for x = {}: {}".format(
                    method, xval, yvals))

            new_y.append(np.mean(yvals))
            if len(idxs) > 1:
                stds.append(np.std(yvals, ddof=1))  # sample std
            else:
                stds.append(0)
        x = new_x
        y = np.array(new_y)
        stds = np.array(stds)
        # x, y = zip(*new_pairs)

        if semilogx:
            x = np.log2(x)

        std_col = {'dTop1': 'stdTop1', 'Top1': 'stdTop1',
                   'dTop5': 'stdTop5', 'Top5': 'stdTop5'}[ycol]
        df_stds = subdf[std_col].values * 100

        stds = np.nanmax(np.vstack([stds, df_stds]), axis=0)

        # ------------------------ compute color/marker and plot stuff

        try:
            verbose = method in ('Magnitude', 'SparseVD')
            if verbose:
                print('trying to get year for method: ', method)
            year = subdf['Year'].values[0]
            # if verbose:
            # print("succeeded! year = '{}'".format(year))
            color = val2color[year]
        except IndexError: # TODO fix not having year for goog results
            print('method: ', method)
            print("failed to get year!")
            print("whole subdf:")
            print(subdf)
            color = None
        # color = None
        marker = val2marker[method]

        # if y.min() < -20:
        #     print('outlier in method: ', method)

        # print("len(x), len(y) = ", len(x), len(y))

        # method = subdf['Method']
        # ax.plot(x, y, label=method, marker=marker, color=color, markersize=6)
        yerr = stds if np.max(stds) > 0 else None
        ax.errorbar(x, y, label=method, marker=marker, color=color, markersize=6,
                    yerr=yerr)

    # assert given_ax
    if given_ax:  # assume func supplying axis will save, show, etc
        return

    if save:
        # path = FIGS_SAVE_DIR / fname
        fname = '{}_{}_{}_{}'.format(dset, model, xcol, ycol)
        save_fig(fname)
    else:
        plt.show()
    plt.close()


# valid in the sense that we've populated x and y data for at least one
# tradeoff curve with this method
def _load_valid_pruning_results(
        combos=None, use_xcols=None, use_ycols=None, min_acc=.5):

    # ------------------------ config

    df = load_results(include_pruning=True, rm_reimplementations=True)

    if combos is None:
        combos = ALL_VALID_DSET_MODEL_COMBOS
    valid_dsets = [combo[0] for combo in combos]
    valid_models = [combo[1] for combo in combos]

    if use_xcols is None:
        use_xcols = ['Theoretical Speedup', 'Compression Ratio']
    if use_ycols is None:
        use_ycols = ['dTop1', 'dTop5']

    # ------------------------ data cleaning

    # rm methods we don't want to plot because they aren't really new
    # methods or baselines that someone proposed (also rm missing method)
    df = df.loc[~df['Method'].isin([METHOD_RANDOM, 'Frankle 2019'])]
    df = df.loc[~df['Method'].isna()]

    # narrow df down to results that will actually get plotted
    df = df.loc[df['Dataset'].isin(valid_dsets)]
    df = df.loc[df['Model'].isin(valid_models)]

    # rm rows that don't have reported top1 or top5 acc below min_acc (this
    # should only affect the state-of-sparsity results, which get crazy
    # speedups on some trials but do no better than chance)
    df = df.loc[~(df['Top1'] < min_acc) | (df['Top5'] < min_acc)]

    # rm rows with nan vals for xcol or ycol; we keep a row if there's at
    # least one (xcol, ycol) combination where it has values for both; we
    # do this by tracking whether each row is valid, and then ORing together
    # the results for every combination
    masks = []
    for xcol in use_xcols:
        for ycol in use_ycols:
            valid_mask = ~(df[xcol].isna() | df[ycol].isna())
            masks.append(valid_mask.values)
    masks = np.vstack(masks)
    keep_mask = masks.sum(axis=0) > 0  # logical OR of vals within each column
    df = df.loc[keep_mask]

    # df = df.loc[df['Method'] != 'Gale 2019, Magnitude']

    df.sort_values(['Year', 'Method'], axis=0, inplace=True)

    return df


def make_pruning_grid_fig(save=True, min_acc=.5):

    combos = [('ImageNet', 'VGG-16'),
              ('ImageNet', 'Alex/CaffeNet'),
              # ('ImageNet', 'ResNet-18'),
              # ('ImageNet', 'ResNet-34'),
              ('ImageNet', 'ResNet-50'),
              ('CIFAR-10', 'ResNet-56')]
    df = _load_valid_pruning_results(combos=combos)
    use_xcols = ['Compression Ratio', 'Theoretical Speedup']
    use_ycols = ['dTop1', 'dTop5']

    # ------------------------ actual plotting

    ncombos = len(combos)
    # fig, axes = plt.subplots(4, ncombos, figsize=(12, 12))
    fig, axes = plt.subplots(4, ncombos, figsize=(12, 11.5))

    val2color, val2marker, _ = make_line_traits_dicts(
            df, color_colname='Year', marker_colname='Method')

    for j, combo in enumerate(combos):
        dset, model = combo
        for ix, xcol in enumerate(use_xcols):
            for iy, ycol in enumerate(use_ycols):
                i = 2 * ix + iy
                ax = axes[i, j]
                assert ax is not None
                plot_tradeoff_curve(df=df, dset=dset, model=model,
                                    xcol=xcol, ycol=ycol, ax=ax,
                                    val2color=val2color, val2marker=val2marker,
                                    semilogx=xcol.startswith('Compression Ratio'))

                ylabel = {'dTop1': 'Change in\nTop-1 Accuracy (%)',
                          'dTop5': 'Change in\nTop-5 Accuracy (%)',
                          'Top1': 'Top-1 Accuracy (%)',
                          'Top5': 'Top-5 Accuracy (%)'}[ycol]
                xlabel = {'Compression Ratio': 'Log2(Compression Ratio)',
                          'Theoretical Speedup': 'Theoretical Speedup'}[xcol]

                ax.set_xlabel(None)
                if i == 0:
                    ax.set_title('{} on {}'.format(model, dset), fontsize=14)
                else:
                    ax.set_title(None)
                if j == 0:
                    ax.set_ylabel(ylabel, fontsize=12)
                else:
                    ax.set_ylabel(None)
                ax.set_xlabel(xlabel, labelpad=1.2)
                # ax.set_xlabel(xlabel)

    # create legend; need union of lines from all plots, which makes
    # life hard
    label2handle = {}
    for ax in axes.ravel():
        handles, labels = ax.get_legend_handles_labels()
        label2handle.update(dict(zip(labels, handles)))

    def sort_key(label):
        text = label[0]

        # # total hck to handle google state-of-sparsity results
        # if text == 'Magnitude':
        #     return '2017 ' + text
        # elif

        year_start_idx = text.find('20')
        year = text[year_start_idx:year_start_idx+4]
        # year = text.split(',')[0].split()[-1]
        return year + text  # rest of label starts with author last name

    # print("label2handle.items()")
    # pprint.pprint(label2handle.items())
    # import sys; sys.exit()

    pairs = sorted(list(label2handle.items()), key=sort_key)
    labels, handles = zip(*pairs)

    # plt.figlegend(handles, labels, loc='lower center', ncol=5)
    plt.figlegend(handles, labels, loc='lower center', ncol=6)

    fig.delaxes(axes[1, 3])
    fig.delaxes(axes[3, 3])

    plt.tight_layout()
    # plt.subplots_adjust(bottom=.2, hspace=.3)
    # plt.subplots_adjust(bottom=.17, hspace=.3)
    plt.subplots_adjust(bottom=.18, hspace=.3)
    # plt.subplots_adjust(bottom=.18)
    if save:
        save_fig('all_pruning_curves')
    else:
        plt.show()
    plt.close()


def make_nresults_fig(save=True, include_mnist=False):
    df = load_results(include_pruning=True)
    df = df.loc[df['Model'] != 'Alex/CaffeNet']
    if not include_mnist:
        df = df.loc[df['Dataset'] != 'MNIST']

    print("total number of results: ", df.shape[0])

    paper2combos = _load_paper2combos_dict(include_mnist=include_mnist)
    paper2numcombos = {k: len(paper2combos[k]) for k in paper2combos}

    id2published = _load_is_published_at_top_conf_dict()
    paper2numcombos_pub = {k: len(paper2combos[k])
                           for k in paper2combos if id2published[k]}
    paper2numcombos_unpub = {k: len(paper2combos[k])
                             for k in paper2combos if not id2published[k]}

    uniq_papers = df['Paper'].unique()

    # paper2numcombos = {}
    # for paper in paper2combos:
    #     combos = paper2combos[paper]
    #     paper2numcombos[paper] = len(combos)
    #     paper2numdsets[paper] = len(combos)

    nresults_dicts = []
    # wait, do I even need paper2numdsets and paper2nummodels?
    paper2numdsets = {}
    paper2nummodels = {}
    for paper in uniq_papers:
        pub = id2published[paper]

        subdf = df.loc[df['Paper'] == paper]
        paper2numdsets[paper] = len(subdf['Dataset'].unique())
        paper2nummodels[paper] = len(subdf['Model'].unique())

        # compute number of rows with each unique (paper, model, dset) combo
        datasets = list(subdf['Dataset'])
        models = list(subdf['Model'])
        combos = list(set(list(zip(datasets, models))))  # probably excessive
        # print("paper: ", paper)
        # print("combos: ", combos)
        # for dset, model in combos:
        for dset, model in ALL_VALID_DSET_MODEL_COMBOS:
            resdf = subdf.loc[
                (subdf['Dataset'] == dset) & (subdf['Model'] == model)]
            # print("resdf shape: ", resdf.shape[0])
            d = {'Paper': paper, 'Published': pub,
                 'Dataset': dset, 'Model': model, 'Nresults': resdf.shape[0]}
            nresults_dicts.append(d)

    nresults_df = pd.DataFrame.from_records(nresults_dicts)

    # ------------------------ plot stuff  # TODO decouple from data munging

    all_ncombos = np.array(list(paper2numcombos.values()))
    all_nresults = nresults_df['Nresults'].values

    # sb.set_style('ticks')  # grids don't work when CDF present
    # sb.set_style('white')  # grids don't work when CDF present
    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_2x1)

    split_by_venue = True
    stacked = True

    title_fontsize = 14
    fontsize = 12.5
    axes[0].set_title('Number of (Dataset, Architecture) Pairs Used',
                      fontsize=title_fontsize)
    axes[0].set_xlabel('Number of pairs', fontsize=fontsize)
    axes[0].set_ylabel('Number of papers\nusing this many pairs',
                       fontsize=fontsize)
    print("number of (dset, model) combos summed across papers: ",
          len(all_ncombos))

    axes[1].set_title('Number of Points used to Characterize Tradeoff Curve',
                      fontsize=title_fontsize)
    axes[1].set_xlabel('Number of points', fontsize=fontsize)
    axes[1].set_ylabel('Number of curves\nusing this many points',
                       fontsize=fontsize)
    print("total number of results characterizing curves: ", len(all_nresults))

    colors = [color_for_idx(1), color_for_idx(0)]

    if split_by_venue:
        # id2published = _load_is_published_at_top_conf_dict()
        # is_pub = [id2published[pid] for pid in uniq_papers]
        all_ncombos_pub = list(paper2numcombos_pub.values())
        all_ncombos_unpub = list(paper2numcombos_unpub.values())

        mask = nresults_df['Published']
        all_nresults_pub = nresults_df['Nresults'].loc[mask]
        all_nresults_unpub = nresults_df['Nresults'].loc[~mask]

        hist_integers(ax=axes[0], ints=[all_ncombos_pub, all_ncombos_unpub],
                      allow_zero=False, label=PUBLISHED_VS_UNPUB_LABELS,
                      stacked=stacked, color=colors)
        hist_integers(ax=axes[1], ints=[all_nresults_pub, all_nresults_unpub],
                      allow_zero=False, label=PUBLISHED_VS_UNPUB_LABELS,
                      stacked=stacked, color=colors)
    else:
        hist_integers(axes[0], all_ncombos, allow_zero=False)
        hist_integers(axes[1], all_nresults, allow_zero=False)
    # ax2.set_ylabel('CDF')

    plt.tight_layout()

    if split_by_venue:
        handles, labels = axes.ravel()[-1].get_legend_handles_labels()
        plt.figlegend(handles, labels, loc='lower center', ncol=2)
        # plt.subplots_adjust(bottom=.11)
        plt.subplots_adjust(bottom=.15)

    if save:
        save_fig('numresults_stats')
    else:
        plt.show()
    plt.close()


def _load_normalized_prune_arch_dfs(normalize_acc=True, normalize_nparams=True,
                                    normalize_nflops=True, **kwargs):
    kwargs.setdefault('include_pruning', True)
    df_prune = load_results(**kwargs)
    df_prune = df_prune.loc[df_prune['Dataset'] == 'ImageNet']
    df_arch = load_existing_models_results()
    # rm date in method name
    df_arch['Method'] = df_arch['Method'].apply(lambda s: s.split(' (')[0])

    # prettier names (and do raw numbers since log scale either way)
    for df in (df_prune, df_arch):
        df['Number of Parameters'] = df['MParams'] * 1e6
        df['Number of FLOPs'] = df['GFLOPS'] * 1e9

    # normalize pruning results to best baseline result for each model
    uniq_models = df_arch['Model'].unique()
    max_top1_for_model = {}
    max_top5_for_model = {}
    median_nparams_for_model = {}
    median_nflops_for_model = {}
    for model in uniq_models:
        subdf = df_arch.loc[df_arch['Model'] == model]
        max_top1_for_model[model] = np.median(subdf['Top1'])
        max_top5_for_model[model] = np.median(subdf['Top5'])
        median_nparams_for_model[model] = np.median(subdf['Number of Parameters'])
        median_nflops_for_model[model] = np.median(subdf['Number of FLOPs'])
    prune_models = df_prune['Model']
    baselines_top1 = np.array([max_top1_for_model.get(model, np.nan)
                               for model in prune_models])
    baselines_top5 = np.array([max_top5_for_model.get(model, np.nan)
                               for model in prune_models])
    baselines_nparams = np.array([median_nparams_for_model.get(model, np.nan)
                                  for model in prune_models])
    baselines_nflops = np.array([median_nflops_for_model.get(model, np.nan)
                                 for model in prune_models])
    changes_top1 = df_prune['dTop1']
    changes_top5 = df_prune['dTop5']
    kept_nparams_frac = df_prune['keptParams']
    kept_nflops_frac = df_prune['keptFLOPS']
    mask1 = ~np.isnan(baselines_top1)
    mask5 = ~np.isnan(baselines_top5)
    mask_nparam = ~np.isnan(baselines_nparams)
    mask_nflops = ~np.isnan(baselines_nflops)
    if normalize_acc:
        orig_top1s = df_prune['Top1'].values.copy()
        orig_top5s = df_prune['Top5'].values.copy()
        top1valid = ~df_prune['Top1'].isna()
        top5valid = ~df_prune['Top5'].isna()
        # new_top1s = (baselines_top1 + changes_top1)[mask1]
        # new_top5s = (baselines_top5 + changes_top5)[mask5]
        df_prune['Top1'].loc[mask1] = (baselines_top1 + changes_top1)[mask1]
        df_prune['Top5'].loc[mask5] = (baselines_top5 + changes_top5)[mask5]

        new_top1s = df_prune['Top1'].values
        new_top5s = df_prune['Top5'].values

        dTop1s = (new_top1s - orig_top1s)[top1valid]
        dTop5s = (new_top5s - orig_top5s)[top5valid]

        # print("dTop1s from normalizing accuracies: ")
        # print(np.sort(dTop1s) * 100)
        # print("dTopts from normalizing accuracies: ")
        # print(np.sort(dTop5s) * 100)

    if normalize_nparams:
        df_prune['Number of Parameters'].loc[mask_nparam] = \
            (baselines_nparams * kept_nparams_frac)[mask_nparam]
    if normalize_nflops:
        df_prune['Number of FLOPs'].loc[mask_nflops] = \
            (baselines_nflops * kept_nflops_frac)[mask_nflops]

    return df_prune, df_arch


def make_pruned_vs_arch_fig(save=True, min_acc=.5, xcol='Compression Ratio',
                            show_norm_effects=False):
    # df_prune, df_arch = _load_normalized_prune_arch_dfs()
    if show_norm_effects:
        # df_prune2, _ = _load_normalized_prune_arch_dfs(normalize_acc=False)
        df_prune, df_arch = _load_normalized_prune_arch_dfs(normalize_acc=False)
    else:
        df_prune, df_arch = _load_normalized_prune_arch_dfs()

    df_vgg_orig = df_arch.loc[df_arch['Method'] == 'VGG']
    df_resnet_orig = df_arch.loc[df_arch['Method'].str.startswith('ResNet')]
    df_effic_orig = df_arch.loc[df_arch['Method'].str.startswith('EfficientNet')]
    df_mobilenet2_orig = df_arch.loc[df_arch['Method'].str.startswith('MobileNet-v2')]

    df_vgg_pruned = df_prune.loc[df_prune['Model'].str.startswith('VGG') & (
        df_prune['Dataset'] == 'ImageNet')]
    df_resnet_pruned = df_prune.loc[
        df_prune['Model'].str.startswith('ResNet') & (
            df_prune['Dataset'] == 'ImageNet')]
    df_mobilenet2_pruned = df_prune.loc[
        df_prune['Model'].str.startswith('MobileNet-v2') & (
            df_prune['Dataset'] == 'ImageNet')]

    df_vgg_pruned['Method'] = 'Pruned VGG Networks'
    df_resnet_pruned['Method'] = 'Pruned ResNets'
    df_mobilenet2_pruned['Method'] = 'Pruned MobileNet-v2'

    # left col = compression ratio, right col = flops
    # upper row = top1, lower row = top5
    fig, axes = plt.subplots(2, 2, figsize=(7, 7), sharex='col', sharey='row')

    # # without normalized accuracy
    # fig2, axes2 = plt.subplots(2, 2, figsize=(7, 7), sharex='col', sharey='row')

    for ax in axes.ravel():
        ax.set_xscale('log')  # NOTE: has to come first or {x,y}lims wrong

    # use this one because adjacent pairs of colors are the same except for
    # opacity (or at least it looks that way), so it's easy to see which
    # pruned results go with which model
    cmap = plt.get_cmap('tab20')

    xcol0 = 'Number of Parameters'
    xcol1 = 'Number of FLOPs'
    ycol0 = 'Top1'
    ycol1 = 'Top5'

    # ------------------------ original architectures
    orig_dfs = (df_vgg_orig, df_resnet_orig,
                df_mobilenet2_orig, df_effic_orig)
    # labels = ('VGG', 'ResNet', 'MobileNet-v2', 'EfficientNet')
    labels = ('VGG (2014)', 'ResNet (2016)', 'MobileNet-v2 (2018)',
              'EfficientNet (2019)')

    arch_color_idxs = np.array([4, 2, 0, 6])
    prune_color_idxs = arch_color_idxs + 1  # tab20 has similar-color pairs
    arch_colors = [cmap(val) for val in (arch_color_idxs / 20 + .01)]
    prune_colors = [cmap(val) for val in (prune_color_idxs / 20 + .01)]

    for i, plt_df in enumerate(orig_dfs):
        # color = cmap((2 * i) / 20 + .01)
        color = arch_colors[i]
        kwargs = dict(color=color, marker='o', label=labels[i])

        # fig with normalized acc
        plt_df = plt_df.sort_values(xcol0, axis=0)
        axes[0, 0].plot(plt_df[xcol0], plt_df[ycol0] * 100, **kwargs)
        axes[1, 0].plot(plt_df[xcol0], plt_df[ycol1] * 100, **kwargs)
        plt_df = plt_df.sort_values(xcol1, axis=0)
        axes[0, 1].plot(plt_df[xcol1], plt_df[ycol0] * 100, **kwargs)
        axes[1, 1].plot(plt_df[xcol1], plt_df[ycol1] * 100, **kwargs)

    # ------------------------ pruned models
    orig_dfs = (df_vgg_pruned, df_resnet_pruned, df_mobilenet2_pruned)
    labels = ('VGG Pruned', 'ResNet Pruned', 'MobileNet-v2 Pruned')

    for i, df in enumerate(orig_dfs):
        # color = cmap((2 * i + 1) / 20 + .01)
        color = prune_colors[i]
        kwargs = dict(color=color, marker='*', label=labels[i])

        # method = list(df['Method'])[0]
        # model = list(df['Model'])[0]

        df = df.sort_values(xcol0, axis=0)
        # print("plotting df for method", method)

        # if model == 'MobileNet-v2': print("sorted mobilenet df:\n", df)

        plt_df = df.loc[~(df[xcol0].isna() | df[ycol0].isna())]
        if plt_df.shape[0] > 0:
            # print("x0 y0")
            # if model == 'MobileNet-v2': print("mobilenet plt_df:\n", plt_df)
            axes[0, 0].scatter(plt_df[xcol0], plt_df[ycol0] * 100, **kwargs)
        plt_df = df.loc[~(df[xcol0].isna() | df[ycol1].isna())]
        if plt_df.shape[0] > 0:
            # print("x0 y1")
            # if model == 'MobileNet-v2': print("mobilenet plt_df:\n", plt_df)
            axes[1, 0].scatter(plt_df[xcol0], plt_df[ycol1] * 100, **kwargs)

        df = df.sort_values(xcol1, axis=0)
        # if model == 'MobileNet-v2': print("sorted mobilenet df:\n", df)
        # print("plotting df for method", list(df['Method'])[0])
        plt_df = df.loc[~(df[xcol1].isna() | df[ycol0].isna())]
        if plt_df.shape[0] > 0:
            # print("x1 y0")
            # if model == 'MobileNet-v2': print("mobilenet plt_df:\n", plt_df)
            axes[0, 1].scatter(plt_df[xcol1], plt_df[ycol0] * 100, **kwargs)
        plt_df = df.loc[~(df[xcol1].isna() | df[ycol1].isna())]
        if plt_df.shape[0] > 0:
            # print("x1 y1")
            # if model == 'MobileNet-v2': print("mobilenet plt_df:\n", plt_df)
            axes[1, 1].scatter(plt_df[xcol1], plt_df[ycol1] * 100, **kwargs)

    # ------------------------ figure stuff

    label2handle = {}
    for ax in axes.ravel():
        handles, labels = ax.get_legend_handles_labels()
        label2handle.update(dict(zip(labels, handles)))
    labels, handles = zip(*label2handle.items())
    # handles, labels = axes[1, 0].get_legend_handles_labels()
    # sort_idxs = np.argsort(labels)[::-1]  # reverse alphabet happens to work
    sort_idxs = np.argsort(labels)
    sort_idxs = np.roll(sort_idxs, -1)  # move efficientnet to end, not start
    labels = np.array(labels)[sort_idxs]
    handles = np.array(handles)[sort_idxs]

    plt.figlegend(handles, labels, loc='lower center', ncol=4)

    fontsize = 14
    axes[0, 0].set_ylabel('Top 1 Accuracy (%)', fontsize=fontsize)
    axes[1, 0].set_ylabel('Top 5 Accuracy (%)', fontsize=fontsize)
    # axes[1, 0].set_xlabel('Millions of Parameters')
    # axes[1, 1].set_xlabel('Billions of FLOPs')
    axes[1, 0].set_xlabel(xcol0, fontsize=fontsize)
    axes[1, 1].set_xlabel(xcol1, fontsize=fontsize)

    plt.suptitle('Speed and Size Tradeoffs for Original and Pruned Models',
                 fontsize=18)

    plt.tight_layout()
    plt.subplots_adjust(top=.92, bottom=.15)

    # print("really small but good models:")
    # # df_small = df_prune.loc[(df_prune[xcol0] < 5e6) & (df_prune['Top1'] > .67)]
    # df_small = df_prune.loc[(df_prune[xcol0] < 12e6) & (df_prune['Top1'] > .67)]
    # print(df_small)

    if show_norm_effects:
        save_fig('arch_vs_prune_unnormed')
    else:
        save_fig('arch_vs_prune')


def make_rn50_magnitude_fig(save=True):
    df_prune, df_arch = _load_normalized_prune_arch_dfs(
        include_pruning=True,
        include_goog_results=True,
        # include_goog_results=False,
        include_lottery_ticket=True,
        rm_reimplementations=False)
    df_prune = df_prune.loc[df_prune['Model'] == 'ResNet-50']
    df_arch = df_arch.loc[df_arch['Model'] == 'ResNet-50']

    # rm goog extreme results that mess up axis scaling
    df_prune = df_prune.loc[df_prune['Top1'] > .65]
    df_prune = df_prune.loc[df_prune['Number of Parameters'] > .99e6]

    magnitude_methods = set([
                    'Gale 2019, Magnitude',
                    'Gale 2019, Magnitude-v2',
                    'Frankle 2019, ResetToEpoch=R',
                    'Frankle 2019, ResetToEpoch=10',
                    'Frankle 2019, PruneAtEpoch=15',
                    'Frankle 2019, PruneAtEpoch=90',
                    'Han 2015',
                    'Han 2016',
                    'Liu 2019, Magnitude'])

    mask = df_prune['Method'].isin(magnitude_methods)
    df_mag = df_prune.loc[mask]
    df_prune = df_prune.loc[~mask]

    # rm other variations of frankle's experiments
    df_prune = df_prune.loc[~df_prune['Method'].str.startswith('Frankle')]
    df_prune = df_prune.loc[df_prune['Method'] != METHOD_RANDOM]

    xcol = 'Number of Parameters'
    ycol = 'Top1'

    fig, axes = plt.subplots(2, 1, figsize=(5.5, 7), sharex=True, sharey=True)

    axes[0].set_title('Pruning ResNet-50 with Unstructured Magnitude-Based Pruning')
    axes[1].set_title('Pruning ResNet-50 with All Other Methods')
    axes[1].set_xlabel('Number of Parameters', labelpad=0, fontsize=10)

    # ax.set_title('Unstructured Magnitude-Based Pruning\n'
    #              'vs All Other Methods for ResNet-50 on ImageNet')
    # ax.set_xlabel('Number of Parameters', labelpad=0)
    # # ax.set_xlabel('Number of Parameters')
    # ax.set_ylabel('Top-1 Accuracy (%)')

    for ax in axes.ravel():
        ax.set_xscale('log')  # NOTE: has to come first or {x,y}lims wrong
        ax.set_ylabel('Top 1 Accuracy (%)', fontsize=12)

    # cmap = plt.get_cmap('tab20')
    # cmap = plt.get_cmap('Set3')
    # cmap_ncolors = 11  # actually 12, but last is ugly neon yellow
    # cmap = plt.get_cmap('tab20c')
    # cmap_ncolors = 17  # last couple are too light / washed out
    # cmap = plt.get_cmap('Dark2')
    # cmap_ncolors = 8
    # cmap = plt.get_cmap('Pastel2')
    # cmap_ncolors = 8
    cmap = plt.get_cmap('Set2')
    cmap_ncolors = 8

    # ------------------------ plot magnitude-based methods
    # # color = 'red'
    # color = color_for_idx(3, cmap='tab10')  # red
    # # color = color_for_idx(0, cmap='tab10')  # blue
    df = df_mag
    marker_idx = 16
    ax = axes[0]
    # for method in sorted(df['Method'].unique()):
    for i, method in enumerate(sorted(df['Method'].unique())):
        color = color_for_idx(i)

        subdf = df.loc[df['Method'] == method]
        # print("subdf:\n", subdf)
        subdf = subdf.sort_values(xcol, axis=0)
        x = subdf[xcol].values
        y = subdf[ycol].values * 100

        marker = marker_for_idx(marker_idx)
        marker_idx += 1

        #  compute std deviation for err bars
        new_x = np.unique(x)
        new_y = np.unique(y)
        new_y = []
        stds = []
        for xval in new_x:
            idxs = np.where(x == xval)[0]
            yvals = y[idxs]

            new_y.append(np.mean(yvals))
            if len(idxs) > 1:
                stds.append(np.std(yvals, ddof=1))  # sample std
            else:
                stds.append(0)
        x = new_x
        y = np.array(new_y)
        stds = np.array(stds)

        # replace above stds if already given in df
        std_col = {'dTop1': 'stdTop1', 'Top1': 'stdTop1',
                   'dTop5': 'stdTop5', 'Top5': 'stdTop5'}[ycol]
        df_stds = subdf.loc[~subdf[std_col].isna()]
        nrows = df_stds.shape[0]
        if nrows > 0:
            # print("method: ", method)
            # print(df_stds[[ycol, std_col]])
            # TODO handle more than easy case of exactly one std for each x
            assert nrows == len(x)
            stds = df_stds[std_col].values * 100

        # ax.scatter(x, y, label=method, color=color, s=ms, marker=marker)
        # ax.plot(x, y, label=method, color=color, ms=ms, marker=marker)
        yerr = stds if np.max(stds) > 0 else None
        if yerr is not None:
            print('plotting error bars for method:', method)
            # print("yerr:\n", yerr)
        # ax.errorbar(x, y, label=method, color=color, ms=ms, marker=marker, yerr=yerr)
        # ax.errorbar(x, y, label=method, color=color, marker=marker, yerr=yerr)

        # split up plot and errorbars because errorbars make the markers
        # in the legend look ugly
        ax.plot(x, y, label=method, color=color, marker=marker)
        ax.errorbar(x, y, label=None, color=color, yerr=yerr, fmt='none')

    # ------------------------ plot non-magnitude methods
    # color = color_for_idx(1)  # blue
    marker_idx = 1
    df = df_prune
    ax = axes[1]
    for i, method in enumerate(sorted(df['Method'].unique())):
        subdf = df.loc[df['Method'] == method]
        # print("subdf:\n", subdf)
        subdf = subdf.sort_values(xcol, axis=0)
        x = subdf[xcol]
        y = subdf[ycol] * 100

        if method == 'Gale 2019, SparseVD':
            ms = 3
        else:
            ms = None

        color = color_for_idx(i + 1)  # shift so sparsevd isn't red
        marker = marker_for_idx(marker_idx)
        marker_idx += 1

        ax.plot(x, y, label=method, color=color, marker=marker,
                ms=ms, linestyle=':')

    all_handles = []
    all_labels = []
    for ax in axes.ravel():
        handles, labels = ax.get_legend_handles_labels()
        all_handles += handles
        all_labels += labels
    handles, labels = all_handles, all_labels
    plt.figlegend(handles, labels, loc='lower center', ncol=2)

    # plt.legend()

    plt.tight_layout()
    # plt.subplots_adjust(bottom=.375)
    plt.subplots_adjust(bottom=.34, hspace=.16)

    if save:
        save_fig('magnitude_vs_nonmagnitude')
    else:
        plt.show()
    plt.close()


# def plot_efficientnet_results(ax=None, save=False, xcol='GFLOPS'):
def plot_efficientnet_results(ax=None, save=False, xcol='MParams'):
    # df = load_efficientnet_results()
    df = load_existing_models_results()

    # ycol = 'Top1'
    ycol = 'Top5'
    dset = 'ImageNet'

    # these are so extreme in some dimension that they mess up the {x,y}lims
    df = df[~df['Method'].isin(
        ['GPipe (2018)', 'AlexNet (2012)', 'SqueezeNet (2016)'])]

    df = df[~df[xcol].isna()]

    given_ax = ax is None
    if given_ax:
        _, ax = plt.subplots(figsize=(10, 8))

    filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    filled_markers += filled_markers

    sb.lineplot(data=df, x=df[xcol], y=df[ycol], estimator=np.median,
                ax=ax, style='Method', hue='Method', dashes=False,
                markers=filled_markers, legend='brief')

    # for i, method in enumerate(list(df['Method'].unique())):
    #     subdf = df[df['Method'] == method]
    #     # if subdf.shape[0] == 1:
    #     #     # plt.scatter(subdf[xcol], subdf[ycol], label=method)
    #     #     ax.scatter(subdf[xcol], subdf[ycol], label=method)
    #     #     continue
    #     marker = ALL_NOT_HIDEOUS_MARKERS[i % len(ALL_NOT_HIDEOUS_MARKERS)]
    #     print("method, marker: ", method, marker)
    #     sb.lineplot(x=subdf[xcol], y=subdf[ycol], estimator=np.median,
    #                 label=method, ax=ax, markers=marker)

    if given_ax:
        plt.legend()
    if 'params' in xcol.lower():
        plt.semilogx()
    plt.title('{} vs Top 1 Accuracy for various architectures on {}'.format(xcol, dset))
    plt.xlabel(xcol)
    plt.ylabel(ycol)

    # print(df)

    if save:
        # path = FIGS_SAVE_DIR / fname
        fname = '{}_{}_{}_{}'.format(dset, model, xcol, ycol)
        save_fig(fname)
    else:
        plt.show()
    plt.close()


# @_memory.cache
def create_improvements_df(save=False):

    combos = VALID_DSET_MODEL_COMBOS_COMBINE_ALEXLIKE

    use_xcols = ['Theoretical Speedup', 'Compression Ratio']
    use_ycols = ['dTop1', 'dTop5']
    full_df = _load_valid_pruning_results(
        combos=combos, use_xcols=use_xcols, use_ycols=use_ycols)

    cmp_dicts = []
    for dset, model in combos:
        combo_df = full_df.loc[full_df['Dataset'] == dset]
        combo_df = combo_df.loc[combo_df['Model'] == model]
        for xcol in use_xcols:
            for ycol in use_ycols:
                df = combo_df
                df = df.loc[~(df[xcol].isna() | df[ycol].isna())]

                groupby = 'Paper'
                # groupby = 'Method'

                uniq_methods = set(df[groupby].unique())
                for method in uniq_methods:
                    subdf = df[df[groupby] == method]
                    # if subdf.shape[0] == 0:
                    #     print("method = ", method)
                    #     print("subdf shape = ", subdf.shape)

                    x = subdf[xcol].values
                    y = subdf[ycol].values
                    points1 = list(zip(x, y))
                    # print("method = ", method)
                    # print("x1 ", x)
                    # print("y1 ", y)
                    # print("subdf shape = ", subdf.shape)
                    # print("len points1 = ", len(points1))

                    other_methods = uniq_methods - set([method])
                    for method2 in other_methods:
                        subdf2 = df[df[groupby] == method2]
                        x2 = subdf2[xcol].values
                        y2 = subdf2[ycol].values
                        points2 = list(zip(x2, y2))

                        # print("method2 = ", method2)
                        # print("x2 ", x2)
                        # print("y2 ", y2)
                        # print("subdf2 shape = ", subdf2.shape)
                        # print("len points2 = ", len(points2))

                        a_curve_algo = 'pareto'
                        b_curve_algo = 'pareto_hull'
                        result = par.compare_curves(points1, points2,
                                                    a_curve_algo=a_curve_algo,
                                                    b_curve_algo=b_curve_algo)

                        npoints = len(result)
                        ndominated = np.sum(result == par.PARETO_DOMINATED)
                        nbelow_curve = np.sum(result == par.BELOW_CURVE)
                        nincomparable = np.sum(result == par.INCOMPARABLE)
                        nabove_curve = np.sum(result == par.ABOVE_CURVE)
                        ndominant = np.sum(result == par.PARETO_DOMINANT)

                        d = {'Paper1': method, 'Paper2': method2,
                             'Dataset': dset, 'Model': model,
                             'CurveAlgo1': a_curve_algo,
                             'CurveAlgo2': b_curve_algo,
                             'NumPoints': npoints,
                             'NumParetoDominated': ndominated,
                             'NumBelowCurve': nbelow_curve,
                             'NumIncomparable': nincomparable,
                             'NumAboveCurve': nabove_curve,
                             'NumParetoDominant': ndominant}
                        cmp_dicts.append(d)

    cmp_df = pd.DataFrame.from_records(cmp_dicts)

    # ------------------------ add col for whether paper1 compares to paper2
    id2baselines = _load_comparesto_dict()

    paper1s = cmp_df['Paper1']
    paper2s = cmp_df['Paper2']
    compares_to = [int(p2 in id2baselines[p1])
                   for p1, p2 in list(zip(paper1s, paper2s))]
    cmp_df['P1ComparesToP2'] = compares_to

    if save:
        cmp_df.to_csv('results/survey/comparison_results.csv')
    return cmp_df


def _load_paper2combos_dict(include_mnist=True):
    results_df = load_results(include_pruning=True)
    results_df = results_df.loc[results_df['Model'] != 'Alex/CaffeNet']
    if not include_mnist:
        results_df = results_df.loc[results_df['Dataset'] != 'MNIST']
    dsets = list(results_df['Dataset'])
    models = list(results_df['Model'])
    # combos = list(zip(dsets, models))
    combos = ["({}, {})".format(dset, model)
              for dset, model in zip(dsets, models)]
    results_df['Combo'] = combos
    paper2combos = {}
    for paper_id in results_df['Paper'].unique():
        # print("loading results for paper ", paper_id)
        # assert paper_id not in (np.nan, None)
        subdf = results_df.loc[results_df['Paper'] == paper_id]
        paper2combos[paper_id] = subdf['Combo'].unique()

    return paper2combos


def _load_comparesto_dict():
    papers_df = load_clean_papers_csv()

    paper_ids = papers_df['Paper']
    compares_to_ids = papers_df['Compares To']

    assert len(paper_ids) == len(compares_to_ids)
    print("number of paper ids: ", len(paper_ids))
    assert len(paper_ids) == 81

    # initialize to empty list for papers that compare to no other methods
    id2baselines = {paper_id: [] for paper_id in paper_ids}

    for paper_id, other_ids in zip(paper_ids, compares_to_ids):
        print(other_ids)
        if isinstance(other_ids, float):  # nan
            continue
        for other_id in other_ids.split(','):
            other_id = other_id.strip()
            # id2baselines.get(paper_id, []).append(other_id)
            id2baselines[paper_id].append(other_id)

    return id2baselines


def _load_is_published_at_top_conf_dict():
    papers_df = load_clean_papers_csv()

    print("papers_df shape: ", papers_df.shape)
    print("paper ids:")
    print("\n".join(sorted(list(papers_df['Paper']))))

    paper_ids = papers_df['Paper']
    # compares_to_ids = papers_df['Compares To']
    # venues = papers_df['Venue'].isin(
    #     # ['NIPS', 'ICML', 'ICLR', 'ECCV', 'CVPR', 'AAAI', 'ICASSP', 'JMLR'])
    #     # top 5 based on https://scholar.google.com/citations?view_op=top_venues&hl=en&vq=eng # noqa
    #     ['NIPS', 'ICML', 'ICLR', 'ECCV', 'CVPR'])
    is_pub = ~papers_df['Venue'].isin(
        # ['NIPS', 'ICML', 'ICLR', 'ECCV', 'CVPR', 'AAAI', 'ICASSP', 'JMLR'])
        # top 5 based on https://scholar.google.com/citations?view_op=top_venues&hl=en&vq=eng # noqa
        ['arXiv', 'OpenReview', 'CVPR Workshops'])
    return dict(zip(paper_ids, is_pub))


def make_numeric_comparisons_fig(save=True, verbose=1, use_convex_hull=True):
    # cmp_df = pd.DataFrame.from_records(cmp_dicts, save=True)
    df = create_improvements_df()
    # df.drop(['CurveAlgo1', 'CurveAlgo2', 'Dataset'], axis=1, inplace=True)
    df.drop(['CurveAlgo1', 'CurveAlgo2'], axis=1, inplace=True)
    # print(df)
    # print(df.loc[(df['Dataset'] == 'ImageNet') & (df['Model'] == 'VGG-16')])
    # print(df.loc[df['Model'] == 'VGG-16'])

    if use_convex_hull:
        df['NumBetter'] = df['NumParetoDominant'].values + df['NumAboveCurve'].values
        df['NumWorse'] = df['NumParetoDominated'].values + df['NumBelowCurve'].values
    else:
        df['NumBetter'] = df['NumParetoDominant']
        df['NumWorse'] = df['NumParetoDominated']

    paper1s = list(df['Paper1'])
    paper2s = list(df['Paper2'])
    used_papers = np.unique(paper1s + paper2s)
    # uniq_paper1 = df['Paper1'].unique()
    # uniq_paper2 = df['Paper2'].unique()

    # ------------------------ number of improvements given compares
    subdf = df.loc[df['P1ComparesToP2'] > 0]
    ncompares = subdf.shape[0]
    dominant_counts = subdf['NumParetoDominant'].values
    ndominant = np.sum(dominant_counts > 0)
    dominated_counts = subdf['NumParetoDominated'].values
    ndominated = np.sum(dominated_counts > 0)

    above_counts = subdf['NumAboveCurve'].values
    nabove = np.sum(above_counts > 0)
    below_counts = subdf['NumBelowCurve'].values
    nbelow = np.sum(below_counts > 0)

    # prints:
    # 68 dominant out of 136 comparisons -> 0.5
    # 25 dominated out of 136 comparisons -> 0.18382352941176472
    if verbose > 0:
        print("{} dominant out of {} comparisons -> {}".format(
            ndominant, ncompares, ndominant / ncompares))
        print("{} dominated out of {} comparisons -> {}".format(
            ndominated, ncompares, ndominated / ncompares))
        print("{} above curve (but not dominant) out of {} comparisons "
              "-> {}".format(nabove, ncompares, nabove / ncompares))
        print("{} below curve (but not dominated) out of {} comparisons "
              "-> {}".format(nbelow, ncompares, nbelow / ncompares))

    # ------------------------ how many papers ever beat a given paper
    # paper2ndominators = {}

    paper2ndominators = {paper: 0 for paper in used_papers}
    for paper in used_papers:
        subdf = df.loc[df['Paper2'] == paper]
        # nrows = subdf.shape[0]
        # ndominated = (subdf['NumParetoDominated'] > 0).sum()
        subdf = subdf.loc[subdf['NumBetter'] > 0]
        ndominating_papers = len(subdf['Paper1'].unique())
        paper2ndominators[paper] = ndominating_papers

    if verbose > 1:
        print("number of papers that beat this paper:")
        pprint.pprint(paper2ndominators)
    all_ndominators = np.array(list(paper2ndominators.values()))
    # axes[0, 0].hist(all_ndominators)

    # ------------------------ how many papers a given paper ever beats
    paper2ndominatees = {paper: 0 for paper in used_papers}
    for paper in used_papers:
        subdf = df.loc[df['Paper1'] == paper]
        # nrows = subdf.shape[0]
        # ndominated = (subdf['NumParetoDominated'] > 0).sum()
        subdf = subdf.loc[subdf['NumBetter'] > 0]
        ndominating_papers = len(subdf['Paper2'].unique())
        paper2ndominatees[paper] = ndominating_papers

    if verbose > 1:
        print("number of papers this paper beats:")
        pprint.pprint(paper2ndominatees)

    all_ndominatees = np.array(list(paper2ndominatees.values()))

    # ------------------------ actual plotting code

    # fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    # sb.set_style('ticks')  # grids don't work when CDF present
    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_2x1)

    split_by_venue = True
    stacked = True

    axes[0].set_title("Number of Papers Beating a Given Paper")
    axes[0].set_xlabel("Number of papers that beat this paper")
    # axes[0].set_ylabel("Number of papers beaten this many times")
    # hist_integers(ints=all_ndominators, ax=axes[0])
    axes[0].set_ylabel("Fraction of papers\nbeaten this many times")

    axes[1].set_title("Number of Papers a Given Paper Beats")
    axes[1].set_xlabel("Number of papers this paper beats")
    # axes[1].set_ylabel("Number of papers that beat this many others")
    # hist_integers(ints=all_ndominatees, ax=axes[1])
    axes[1].set_ylabel("Fraction of papers\nthat beat this many others")

    if split_by_venue:
        id2published = _load_is_published_at_top_conf_dict()

        # all_ndominators = np.array(list(paper2ndominators.values()))
        # all_ndominatees = np.array(list(paper2ndominatees.values()))

        all_ndominators_pub = [paper2ndominators[pid] for pid in
                               paper2ndominators if id2published[pid]]
        all_ndominators_unpub = [paper2ndominators[pid] for pid in
                                 paper2ndominators if not id2published[pid]]
        all_ndominatees_pub = [paper2ndominatees[pid] for pid in
                               paper2ndominatees if id2published[pid]]
        all_ndominatees_unpub = [paper2ndominatees[pid] for pid in
                                 paper2ndominatees if not id2published[pid]]

        hist_integers(ints=[all_ndominators_pub, all_ndominators_unpub],
                      ax=axes[0], label=PUBLISHED_VS_UNPUB_LABELS,
                      stacked=stacked, density=True)
        hist_integers(ints=[all_ndominatees_pub, all_ndominatees_unpub],
                      ax=axes[1], label=PUBLISHED_VS_UNPUB_LABELS,
                      stacked=stacked, density=True)
    else:
        hist_integers(ints=all_ndominators, ax=axes[0], density=True)
        hist_integers(ints=all_ndominatees, ax=axes[1], density=True)

    plt.tight_layout()

    if split_by_venue:
        handles, labels = axes.ravel()[-1].get_legend_handles_labels()
        plt.figlegend(handles, labels, loc='lower center', ncol=2)
        # plt.subplots_adjust(bottom=.1)
        plt.subplots_adjust(bottom=.15)

    if save:
        save_fig('comparisons_hists')
    else:
        plt.show()
    plt.close()


def print_misc_results_stats():
    # load results for all valid combos
    df = load_results(include_pruning=True)
    valid_combos = ALL_VALID_DSET_MODEL_COMBOS
    # valid_combos = VALID_DSET_MODEL_COMBOS_COMBINE_ALEXLIKE

    valid_combo_strs = set(['{}, {}' .format(dset, model)
                            for dset, model in valid_combos])

    # df = df.loc[df['Model'] != 'Alex/CaffeNet']
    df = df.loc[~df['Model'].isin(('Alex/CaffeNet', '???'))]
    df = df.loc[df['Dataset'] != '???']

    # orig_df = df[['Paper', 'Dataset', 'Model', 'dTop1', 'dTop5',
    #          'Compression Ratio', 'Theoretical Speedup']].drop_duplicates()

    df = df[['Paper', 'Dataset', 'Model']].drop_duplicates()

    # df = df.loc[df['Dataset'] != 'MNIST']
    dsets = df['Dataset']
    models = df['Model']
    combos = ['{}, {}' .format(dset, model)
              for dset, model in zip(dsets, models)]
    df['Combo'] = combos
    uniq_combos, counts = np.unique(combos, return_counts=True)
    sort_idxs = np.argsort(counts)[::-1]
    uniq_combos = np.array(uniq_combos)[sort_idxs]
    counts = np.array(counts)[sort_idxs]
    pairs = list(zip(uniq_combos, counts))
    pairs_not_mnist = [(combo, count) for combo, count in pairs if not combo.startswith('MNIST')]
    counts_not_mnist = np.array([pair[1] for pair in pairs_not_mnist])
    # print("number of papers: ", papers_df.shape[0])
    print("number of models: ", len(df['Model'].unique()))
    print("number of datasets: ", len(df['Dataset'].unique()))
    print("number of unique combos: ", len(uniq_combos))
    print("number of combos used once: ", np.sum(counts == 1))
    print("number of combos used more than once: ", np.sum(counts > 1))
    print("sum of counts where count > 1:", np.sum((counts > 1) * counts))
    print("sum of counts where count > 1 excluding MNIST:",
          np.sum((counts_not_mnist > 1) * counts_not_mnist))
    print("sum of (counts - 1):", np.sum(counts - 1))
    print("how many comparison results I have:")
    print(sum([count for combo, count in pairs if combo in valid_combo_strs]))
    print("how many more results I'd need to get all the comparisons, excluding MNIST:")
    print(sum([count for combo, count in pairs_not_mnist
               if (count > 1) and (combo not in valid_combo_strs)]))
    print("Combos and how many papers use them:")
    for combo, count in pairs:
        if count > 1:
            print("{} & {} \\\\".format(combo, count))


def print_misc_papers_stats():
    df = load_clean_papers_csv()
    print("number of papers: ", df.shape[0])

    # venues table
    uniq_venues = df['Venue'].unique()
    counts = [(df['Venue'] == venue).sum() for venue in uniq_venues]
    venue_count = sorted(list(zip(uniq_venues, counts)),
                         key=lambda tup: tup[1], reverse=True)
    venue_count = ['{} & {} \\\\'.format(*entry) for entry in venue_count]
    print("venues:")
    print("\n".join(venue_count))


def generate_bibtex_with_citekeys():
    df = load_clean_papers_csv(drop_bibtex=False)
    names = df['Paper']
    bibtexs = df['BibTeX']

    bibtex_objects = [bib.loads(bibtex_str.strip('').strip("'"))
                      for bibtex_str in bibtexs]

    s = ''

    for obj, name in zip(bibtex_objects, names):
        # this is remarkably circuitous; need to assign to entries property,
        # not just modify the dict, for changes to take effect
        entries_dict = obj.entries[0]
        entries_dict['ID'] = name
        obj.entries = [entries_dict]
        s += bib.dumps(obj)

    # print("================================ resulting bibtex: ")
    # print(s)

    with open('prune.bib', 'w') as bibfile:
        bibfile.write(s)


if __name__ == '__main__':
    make_pruning_grid_fig()
    make_papers_comparisons_fig()
    make_nresults_fig()
    make_numeric_comparisons_fig()
    make_rn50_magnitude_fig()
    make_pruned_vs_arch_fig()
    # make_pruned_vs_arch_fig(show_norm_effects=True)  # not in paper

    estimated_model_stats()
    print_misc_results_stats()
    print_misc_papers_stats()
    generate_bibtex_with_citekeys()
