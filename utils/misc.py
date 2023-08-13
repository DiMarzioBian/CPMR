import numpy as np
import torch.nn as nn


def cal_mrr(rank_u):
    rank_u = np.array(rank_u)
    return (1 / rank_u).mean()


def cal_recall(rank_u, k):
    rank_u = np.array(rank_u)
    return (rank_u <= k).mean()


def init_linear(fc_layer):
    """ initialize linear layer parameters """
    nn.init.eye_(fc_layer.weight)  # eye_ converges a little bit faster than xavier_normal_
    if fc_layer.bias is not None:
        nn.init.zeros_(fc_layer.bias)


def init_embedding(emb_layer):
    """ initialize embedding layer via truncated normalization """
    nn.init.trunc_normal_(emb_layer.weight)
