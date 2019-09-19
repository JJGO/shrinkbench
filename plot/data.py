import json
import pathlib
import string
import pandas as pd

COLUMNS = ['dataset', 'model', 'strategy', 'compression',
           'pre_acc1', 'pre_acc5', 'post_acc1', 'post_acc5',
           'size', 'size_nz', 'real_compression',
           'memory', 'memory_nz',
           'flops', 'flops_nz', 'speedup',
           'completed', 'seed',
           'batch_size', 'epochs', 'optim', 'lr',
           'resume',
           'path', 'completed_epochs']


def nickname(x):
    return "".join(c for c in x if c not in string.ascii_lowercase)


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
            if params['train_kwargs']['epochs'] > 0:
                row += [len(finetuning)]
            else:
                row += [0]
            results.append(row)

    df = pd.DataFrame(data=results, columns=COLUMNS)
    df.dataset.replace('ImageNetHDF5', 'ImageNet', inplace=True)
    df = broadcast_unitary_compression(df)
    df['strategy_'] = df.strategy.map(lambda x: 'TRAIN' if x is None else nickname(x))
    df = df.sort_values(by=['dataset', 'model', 'strategy', 'compression', 'seed'])
    return df


def df_filter(df, **kwargs):

    for k, vs in kwargs.items():
        # selector = pd.Series(np.zeros(len(df)), dtype=bool)
        if not isinstance(vs, list):
            vs = [vs]
        df = df[getattr(df, k).isin(vs)]
        # for v in vs:
        #     selector |= (df == v)
        # df = df[selector]
    return df


def broadcast_unitary_compression(df):
    for _, row in df[df['compression'] == 1].iterrows():
        for strategy in set(df['strategy']):
            if strategy is not None:
                new_row = row.copy()
                new_row['strategy'] = strategy
                df = df.append(new_row, ignore_index=True)
    return df
