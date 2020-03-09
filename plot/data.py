import json
import pathlib
import string
import pandas as pd

COLUMNS = ['dataset', 'model',
           'strategy', 'compression',
           'size', 'size_nz', 'real_compression',
           'flops', 'flops_nz', 'speedup',
           'pre_acc1', 'pre_acc5', 'post_acc1', 'post_acc5',
           'seed', 'batch_size', 'epochs', 'optim', 'lr',
           'completed_epochs', 'path']


def df_from_results(results_path, glob='*'):
    results = []
    results_path = pathlib.Path(results_path)

    for exp in results_path.glob(glob):
        with open(exp / 'params.json', 'r') as f:
            params = eval(json.load(f)['params'])
        with open(exp / 'metrics.json', 'r') as f:
            metrics = json.load(f)
        logs = pd.read_csv(exp / 'logs.csv')

        row = [
            # Params
            params['dataset'],
            params['model'],
            params['strategy'],
            params['compression'],
            # Metrics
            metrics['size'],
            metrics['size_nz'],
            metrics['compression_ratio'],
            metrics['flops'],
            metrics['flops_nz'],
            metrics['theoretical_speedup'],
            # Pre Accs
            metrics['val_acc1'],
            metrics['val_acc5'],
            # Post Accs
            logs['val_acc1'].max(),
            logs['val_acc5'].max(),
            # Other params
            params['seed'],
            params['dl_kwargs']['batch_size'],
            params['train_kwargs']['epochs'],
            params['train_kwargs']['optim'],
            params['train_kwargs']['lr'],
            len(logs), #Completed epochs
            str(exp),
        ]
        results.append(row)

    df = pd.DataFrame(data=results, columns=COLUMNS)
    df = broadcast_unitary_compression(df)
    df = df.sort_values(by=['dataset', 'model', 'strategy', 'compression', 'seed'])
    return df


def df_filter(df, **kwargs):

    for k, vs in kwargs.items():
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
