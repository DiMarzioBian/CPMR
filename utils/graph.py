import numpy as np
from scipy import sparse as sp
import torch


def inverse_degree_array(d):
    with np.errstate(divide='ignore'):
        d_inv = 1. / d
    d_inv[np.isinf(d_inv)] = 0.
    return d_inv


def bi_adj_to_laplacian(mat_b):
    """ convert ui matrix to propagation matrix, in Chapter 4.1 """
    m, n = mat_b.shape
    A = sp.vstack([sp.hstack([sp.csc_matrix((m, m)), mat_b]), sp.hstack([mat_b.T, sp.csc_matrix((n, n))])])
    d = np.array(A.sum(0)).squeeze()
    sqrt_inv_d = inverse_degree_array(d) ** .5
    diag_mat = sp.diags(sqrt_inv_d)
    return (diag_mat @ A @ diag_mat + sp.eye(m + n)) / 2


def bi_adj_to_propagation(mat_b):
    """ convert matrix into one direction propagation matrix """
    d_item = np.array(mat_b.sum(0)).squeeze()
    d_user = np.array(mat_b.sum(1)).squeeze()

    d_item_inv = inverse_degree_array(d_item)
    d_user_inv = inverse_degree_array(d_user)

    return sp.diags(d_user_inv) @ mat_b, sp.diags(d_item_inv) @ mat_b.T


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """ convert a scipy sparse matrix to a torch sparse tensor,
        i.e., convert u-i matrix to a square matrix """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)
