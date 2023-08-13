import torch
from torch import nn
from torch.nn import functional as F

from models.CGNN import CGNN
from utils.misc import init_linear


class UpdateModule(nn.Module):
    """ update from t_minus to t_plus """
    def __init__(self, d_latent):
        super().__init__()
        self.fc_uu = nn.Linear(d_latent, d_latent)
        self.fc_ii = nn.Linear(d_latent, d_latent)
        self.fc_ui = nn.Linear(d_latent, d_latent, bias=False)
        self.fc_iu = nn.Linear(d_latent, d_latent, bias=False)
        for layer in (self.fc_uu, self.fc_ii, self.fc_ui, self.fc_iu):
            init_linear(layer)

    def forward(self, xu_t, xi_t, adj_tgt_i2u, adj_tgt_u2i):

        delta_u = F.relu(self.fc_uu(xu_t) + adj_tgt_i2u @ self.fc_iu(xi_t))
        delta_i = F.relu(self.fc_ii(xi_t) + adj_tgt_u2i @ self.fc_ui(xu_t))

        mask_u = (torch.sparse.sum(adj_tgt_i2u, 1).to_dense() > 0).float()
        mask_i = (torch.sparse.sum(adj_tgt_u2i, 1).to_dense() > 0).float()

        delta_u = delta_u * mask_u.unsqueeze(1)
        delta_i = delta_i * mask_i.unsqueeze(1)

        return delta_u, delta_i


class EvolutionModule(nn.Module):
    """ evolve from t_plus to t_minus """
    def __init__(self, args):
        super().__init__()
        self.n_user = args.n_user
        self.n_item = args.n_item

        self.gnn = CGNN(args)

    def forward(self, adj_his, t_diff, xu_t_plus, xi_t_plus, xu_embed, xi_embed):
        x_t_plus = torch.cat([xu_t_plus, xi_t_plus], 0)
        x_embed = torch.cat([xu_embed, xi_embed], 0)

        norm_ = torch.norm(x_t_plus, dim=1).max()

        x_t_plus = x_t_plus / norm_
        x_embed = x_embed / norm_

        x_gnn = self.gnn(x_embed, x_t_plus, t_diff, adj_his)
        xu_t_minus, xi_t_minus = torch.split(x_gnn, [self.n_user, self.n_item], 0)
        return xu_t_minus, xi_t_minus


class PredictModule(nn.Module):
    """ predict from x_t_minus to prediction """
    def __init__(self, d_input, d_latent):
        super().__init__()
        self.u_pred_mapping = nn.Linear(d_input, d_latent)
        self.i_pred_mapping = nn.Linear(d_input, d_latent)
        for layer in (self.u_pred_mapping, self.i_pred_mapping):
            init_linear(layer)

    def forward(self, zu, zi):
        zu = self.u_pred_mapping(zu)
        zi = self.i_pred_mapping(zi)
        return (zu * zi).sum(dim=-1)
