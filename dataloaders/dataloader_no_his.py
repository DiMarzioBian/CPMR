import os
import numpy as np
import pandas as pd
from scipy import sparse as sp
import torch
from prefetch_generator import BackgroundGenerator

from utils.graph import bi_adj_to_laplacian, bi_adj_to_propagation, sparse_mx_to_torch_sparse_tensor


class Dataset:
    def __init__(self, args, df, df_0=None):
        self.n_user = args.n_user
        self.n_item = args.n_item
        self.n_neg_sample = args.n_neg_sample
        self.len_ctx = args.len_ctx
        self.ts_max = args.ts_max
        self.idx_pad = args.idx_pad

        self.all_i = pd.Series(range(1, self.n_item))  # remove padding 0
        self.all_u = pd.Series(range(1, self.n_user))
        self.df = df

        # get index of unique timestamps
        self.ts_unique = np.unique(df['time'])
        self.n_ts = len(self.ts_unique)
        end_idx_ts_dict = {t: i + 1 for i, t in enumerate(df['time'])}

        # inherit previous interactions
        self.t0 = self.df['time'].min()
        if df_0 is not None:
            len_df_0 = df_0.shape[0]
            self.cum_n_records = np.array([len_df_0] + [end_idx_ts_dict[t] + len_df_0 for t in self.ts_unique])
            self.df = pd.concat([df_0, self.df])
        else:
            self.cum_n_records = np.array([0] + [end_idx_ts_dict[t] for t in self.ts_unique])
            self.df = df

    def __len__(self):
        return self.n_ts

    def __get_item__(self, idx):
        return self.get_item(idx)

    def get_adj_ui(self, df):
        return sp.csc_matrix((np.ones(len(df)), (df.iloc[:, 0], df.iloc[:, 1])), shape=[self.n_user, self.n_item])

    def get_ctx_graph(self, ts_now, idx_tgt_start):
        idx_ctx_start = self.cum_n_records[(self.ts_unique < (ts_now - self.len_ctx)).sum()]
        df_ctx = self.df.iloc[idx_ctx_start:idx_tgt_start]
        return self.get_adj_ui(df_ctx)

    def get_item(self, idx):
        ts_now = self.ts_unique[idx]
        ts_inc = (ts_now - (self.ts_unique[idx - 1] if idx > 0 else self.t0)) / self.ts_max

        idx_tgt_start = self.cum_n_records[idx]
        idx_tgt_end = self.cum_n_records[idx + 1]
        df_his_tgt = self.df.iloc[:idx_tgt_end]  # all happened interactions

        # get instant interactions graph (both positive and negative targets)
        df_tgt = self.df.iloc[idx_tgt_start:idx_tgt_end]
        adj_tgt = self.get_adj_ui(df_tgt)
        tgt_u, tgt_i = df_tgt['user'].values, df_tgt['item'].values

        tgt_u_neg = \
            np.array([self.all_u[~self.all_u.isin(df_his_tgt[df_his_tgt['item'] == i]['user'])].sample(self.n_neg_sample).values
                      for i in tgt_i])
        tgt_i_neg = \
            np.array([self.all_i[~self.all_i.isin(df_his_tgt[df_his_tgt['user'] == u]['item'])].sample(self.n_neg_sample).values
                      for u in tgt_u])
        # assert (self.idx_pad not in tgt_u_neg) and (self.idx_pad not in tgt_i_neg) and (self.idx_pad not in tgt_i)

        adj_ctx = self.get_ctx_graph(ts_now, idx_tgt_start)

        return ts_inc, None, adj_ctx, adj_tgt, tgt_u, tgt_i, tgt_u_neg, tgt_i_neg

    def get_last_df(self):
        """ get last DataFrame for inherent """
        return self.df


class Dataloader:
    def __init__(self, args, ds):
        self.ds = ds
        self.device = args.device
        self.alpha_spectrum = args.alpha_spectrum
        self.n_batch_load = args.n_batch_load
        self.length = len(self.ds)

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.get_iter(0)

    def get_iter(self, start_idx=0):
        return BackgroundGenerator(self._get_iter(start_idx), self.n_batch_load)

    def _get_iter(self, start_idx=0):
        for i in range(start_idx, len(self.ds)):
            ts_inc, _, adj_ctx, adj_tgt, tgt_u, tgt_i, tgt_u_neg, tgt_i_neg = self.ds.get_item(i)

            adj_his_laplacian = None

            adj_ctx_laplacian = sparse_mx_to_torch_sparse_tensor(bi_adj_to_laplacian(adj_ctx) * self.alpha_spectrum
                                                                 ).to(self.device)

            adj_tgt_i2u, adj_tgt_u2i = (sparse_mx_to_torch_sparse_tensor(v).to(self.device)
                                        for v in [*bi_adj_to_propagation(adj_tgt)])

            # gen other tensors
            tgt_u = torch.as_tensor(tgt_u, dtype=torch.long).unsqueeze(-1).to(self.device)
            tgt_i = torch.as_tensor(tgt_i, dtype=torch.long).unsqueeze(-1).to(self.device)
            tgt_u_neg = torch.as_tensor(tgt_u_neg, dtype=torch.long).to(self.device)
            tgt_i_neg = torch.as_tensor(tgt_i_neg, dtype=torch.long).to(self.device)

            yield \
                ts_inc, adj_his_laplacian, adj_ctx_laplacian, adj_tgt_i2u, adj_tgt_u2i, tgt_u, tgt_i, tgt_u_neg, \
                tgt_i_neg


def split_data(args, df):
    """ Split whole data_raw into train, validation and test sets """
    proportion_train = args.proportion_train

    num_interactions = len(df)
    idx_val_start = int(num_interactions * proportion_train)
    idx_test_start = int(num_interactions * (proportion_train + 0.1))
    idx_test_end = int(num_interactions * (proportion_train + 0.2))

    df_tr = df.iloc[:idx_val_start]
    df_val = df.iloc[idx_val_start:idx_test_start]
    df_te = df.iloc[idx_test_start:idx_test_end]

    return df_tr, df_val, df_te


def read_data(args):
    """ check if processed csv, matrix dictionary or raw file exists """
    # check data
    if not os.path.exists(args.f_csv):
        if os.path.exists(args.f_raw):
            raise FileNotFoundError(f'Raw dataset {args.dataset} found without preprocessed, '
                                    f'please run preprocessed.py first.')
        else:
            raise FileNotFoundError(f'Dataset {args.dataset} not found.')

    # load data, add padding
    df = pd.read_csv(args.f_csv)
    df.columns = ['user', 'item', 'time']
    df['item'] = df['item'].add(1)
    args.ts_max = df['time'].max()

    # check dataframe, considering item padding
    assert df['user'].max() + 1 == df['user'].nunique()
    assert df['item'].max() == df['item'].nunique()
    assert (df['time'].diff().iloc[1:] >= 0).all()
    args.n_user, args.n_item = df.iloc[:, :2].max() + 1

    return df


def get_dataloader(args, noter):
    df = read_data(args)
    df_tr, df_val, df_te = split_data(args, df)

    # pack to Dataset
    noter.log_msg(f'\n[info] Dataset')
    ds_tr = Dataset(args, df_tr)
    ds_val = Dataset(args, df_val, df_0=ds_tr.get_last_df())
    ds_te = Dataset(args, df_te, df_0=ds_val.get_last_df())
    noter.log_msg(f'\t| users {args.n_user} | items {args.n_item - 1} | interactions {len(df)} '
                  f'| timestamps {df["time"].nunique()} |'
                  f'\n\t| interactions | train {len(df_tr)} | valid {len(df_val)} | test {len(df_te)} |'
                  f'\n\t| timestamps   | train {len(ds_tr)} | valid {len(ds_val)} | test {len(ds_te)} |')

    # pack to DataLoader
    trainloader = Dataloader(args, ds_tr)
    valloader = Dataloader(args, ds_val)
    testloader = Dataloader(args, ds_te)

    return trainloader, valloader, testloader
