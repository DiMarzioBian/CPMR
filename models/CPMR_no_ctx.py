import torch
from torch import nn
from torch.nn import functional as F

from models.PMTL_ablation import CGC_1_2 as PMTL
from models.Modules import EvolutionModule, UpdateModule, PredictModule
from utils.misc import init_embedding


class CPMR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.n_user = args.n_user
        self.n_item = args.n_item
        self.d_latent = args.d_latent
        self.n_neg_sample = args.n_neg_sample
        self.idx_pad = args.idx_pad

        # build model
        self.embeds_u = nn.Embedding(self.n_user, self.d_latent)
        self.embeds_i = nn.Embedding(self.n_item, self.d_latent)
        for layer in (self.embeds_u, self.embeds_i):
            init_embedding(layer)

        self.evolver = EvolutionModule(args)

        self.fuser_u = PMTL(self.d_latent, self.d_latent)
        self.fuser_i = PMTL(self.d_latent, self.d_latent)

        self.predictor = PredictModule(2 * self.d_latent, self.d_latent)

        self.updater = UpdateModule(self.d_latent)

    def get_init_states(self):
        return self.embeds_u.weight.clone().detach(), self.embeds_i.weight.clone().detach()

    def forward(self, batch, xu_in, xi_in):
        t_diff, adj_his, _, adj_tgt_i2u, adj_tgt_u2i, tgt_u, tgt_i, tgt_u_neg, tgt_i_neg = batch

        # propagate during non-events
        xu_t_minus, xi_t_minus = \
            self.evolver(adj_his, t_diff, xu_in, xi_in, self.embeds_u.weight, self.embeds_i.weight)

        # fuse history and context info
        zu_t, xu_t = self.fuser_u(xu_t_minus)
        zi_t, xi_t = self.fuser_i(xi_t_minus)

        # calculate loss
        zu_enc = torch.hstack((zu_t, self.embeds_u.weight))
        zu_pos = F.embedding(tgt_u, zu_enc)
        zu_neg = F.embedding(tgt_u_neg, zu_enc)

        zi_enc = torch.hstack((zi_t, self.embeds_i.weight))
        zi_pos = F.embedding(tgt_i, zi_enc)
        zi_neg = F.embedding(tgt_i_neg, zi_enc)

        loss_rec = self.cal_loss(zu_pos, zi_pos, zu_neg, zi_neg)

        # update
        dxu_t, dxi_t = self.updater(xu_t, xi_t, adj_tgt_i2u, adj_tgt_u2i)

        xu_t_plus = xu_t + dxu_t
        xi_t_plus = xi_t + dxi_t

        return loss_rec, zu_pos, zi_enc, xu_t_plus, xi_t_plus

    def cal_loss(self, zu_pos, zi_pos, zu_neg, zi_neg):
        pos_scores = self.predictor(zu_pos, zi_pos)

        neg_scores_u = self.predictor(zu_pos, zi_neg)
        neg_scores_i = self.predictor(zu_neg, zi_pos)

        scores = torch.cat([pos_scores, neg_scores_u, neg_scores_i], dim=-1)
        loss = -F.log_softmax(scores, 1)[:, 0].mean()
        return loss
