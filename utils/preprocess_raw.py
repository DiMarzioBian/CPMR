import os
import pandas as pd
import argparse

from utils.constant import MAPPING_RAW, MAPPING_TS_UNIT


def read_amazon(dataset: str, path: str):
    """ Read amazon data_raw from raw data_raw sheet """
    print(f'\n[Info] Start reading dataset {dataset}.')
    df = pd.read_json(path, lines=True)
    df = df.loc[:, ['reviewerID', 'asin', 'unixReviewTime']]
    df.columns = ['user', 'item', 'ts']
    df.sort_values(by=['ts', 'user'], inplace=True)
    print(f'\n[Info] Successfully reading dataset {dataset}.')
    return df


def read_movielens(dataset: str, path: str):
    """ Read MovieLens data_raw from raw data_raw sheet """
    print(f'\n[Info] Start reading dataset {dataset}.')
    df = pd.read_csv(path, header=None, sep='\t', usecols=[0, 1, 3])
    df.columns = ['user', 'item', 'ts']
    df.sort_values(by=['ts', 'user'], inplace=True)
    print(f'\n[Info] Successfully read dataset "{dataset}".')
    return df


def filter_under_5(df: pd.DataFrame):
    """ filter out cold-start item and users appears less than 5 """
    u_5 = df['user'].value_counts()[df['user'].value_counts() >= 5].index
    i_5 = df['item'].value_counts()[df['item'].value_counts() >= 5].index

    if len(df['user'].unique()) == len(u_5) and len(df['item'].unique()) == len(i_5):
        return df
    else:
        df = df[df['user'].isin(u_5)]
        df = df[df['item'].isin(i_5)]
        return filter_under_5(df)


def reindex_data(df: pd.DataFrame, ts_unit: int, to_float=False):
    """ Re-index both users and items """
    map_u, map_v = {}, {}
    list_u, list_v = df['user'].unique().tolist(), df['item'].unique().tolist()

    for i, idx_u in enumerate(list_u):
        map_u[idx_u] = i
    for i, idx_v in enumerate(list_v):
        map_v[idx_v] = i

    df['user'] = [map_u[u] for u in df['user'].tolist()]
    df['item'] = [map_v[v] for v in df['item'].tolist()]
    if to_float:
        df['ts'] = ((df['ts'] - df['ts'].min()) / ts_unit).astype(float)
    else:
        df['ts'] = ((df['ts'] - df['ts'].min()) / ts_unit).astype(int)
    print('\n[info] Dataset contains', df.shape[0], 'interactions,', len(list_u), 'users and', len(list_v), 'items.')

    return df, len(list_u), len(list_v), map_u, map_v


def save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    print('\n[info] CSV save to file:', path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='video')
    args = parser.parse_args()

    (dataset, f_name) = MAPPING_RAW[args.data]
    args.to_float = (dataset in ['ml-100k'])

    # check dataset
    path_raw = f'../data_processed/raw/{f_name}'
    path_csv = f'../data_processed/{dataset}_5.csv'

    if os.path.exists(path_raw):
        try:
            ts_unit = MAPPING_TS_UNIT[dataset]
            print(f'\n[Info] Successfully detect dataset "{dataset}" and set unit timestamp.')
        except KeyError:
            raise KeyError(f'Please set timestamp unit for dataset "{dataset}" in constant.py file.')
    else:
        raise FileNotFoundError(f'Raw dataset "{dataset}" not found in root path.')

    # read and reindex dataset
    if dataset in ['ml-100k']:
        df_raw = read_movielens(dataset, path_raw)
        df_5_raw = filter_under_5(df_raw)
    else:
        df_5_raw = read_amazon(dataset, path_raw)

    # reindex
    df_5, n_user, n_item, *_ = reindex_data(df_5_raw, ts_unit, to_float=args.to_float)

    save_csv(df_5, path_csv)


if __name__ == '__main__':
    main()
