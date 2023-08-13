import torch
from torch import nn
from torch.nn import functional as F

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

        self.evolver_his = EvolutionModule(args)
        self.evolver_ctx = EvolutionModule(args)

        self.predictor = PredictModule(2 * self.d_latent, self.d_latent)

        self.updater_his = UpdateModule(self.d_latent)
        self.updater_ctx = UpdateModule(self.d_latent)

    def get_init_states(self):
        return self.embeds_u.weight.clone().detach(), self.embeds_i.weight.clone().detach(), \
            self.embeds_u.weight.clone().detach(), self.embeds_i.weight.clone().detach()

    def forward(self, batch, xu_in_his, xi_in_his, xu_in_ctx, xi_in_ctx):
        t_diff, adj_his, adj_ctx, adj_tgt_i2u, adj_tgt_u2i, tgt_u, tgt_i, tgt_u_neg, tgt_i_neg = batch

        # propagate during non-events
        xu_t_minus_his, xi_t_minus_his = \
            self.evolver_his(adj_his, t_diff, xu_in_his, xi_in_his, self.embeds_u.weight, self.embeds_i.weight)
        xu_t_minus_ctx, xi_t_minus_ctx =\
            self.evolver_ctx(adj_ctx, t_diff, xu_in_ctx, xi_in_ctx, self.embeds_u.weight, self.embeds_i.weight)

        # fuse history and context info
        # zu_t = torch.hstack((xu_t_minus_his, xu_t_minus_ctx))
        # zi_t = torch.hstack((xi_t_minus_his, xi_t_minus_ctx))
        zu_t = xu_t_minus_his + xu_t_minus_ctx
        zi_t = xi_t_minus_his + xi_t_minus_ctx

        # calculate loss
        zu_enc = torch.hstack((zu_t, self.embeds_u.weight))
        zu_pos = F.embedding(tgt_u, zu_enc)
        zu_neg = F.embedding(tgt_u_neg, zu_enc)

        zi_enc = torch.hstack((zi_t, self.embeds_i.weight))
        zi_pos = F.embedding(tgt_i, zi_enc)
        zi_neg = F.embedding(tgt_i_neg, zi_enc)

        loss_rec = self.cal_loss(zu_pos, zi_pos, zu_neg, zi_neg)

        # update
        dxu_his, dxi_his = self.updater_his(xu_t_minus_his, xi_t_minus_his, adj_tgt_i2u, adj_tgt_u2i)
        dxu_ctx, dxi_ctx = self.updater_ctx(xu_t_minus_ctx, xi_t_minus_ctx, adj_tgt_i2u, adj_tgt_u2i)

        xu_t_plus_his = xu_t_minus_his + dxu_his
        xi_t_plus_his = xi_t_minus_his + dxi_his
        xu_t_plus_ctx = xu_t_minus_ctx + dxu_ctx
        xi_t_plus_ctx = xi_t_minus_ctx + dxi_ctx

        return loss_rec, zu_pos, zi_enc, xu_t_plus_his, xi_t_plus_his, xu_t_plus_ctx, xi_t_plus_ctx

    def cal_loss(self, zu_pos, zi_pos, zu_neg, zi_neg):
        pos_scores = self.predictor(zu_pos, zi_pos)

        neg_scores_u = self.predictor(zu_pos, zi_neg)
        neg_scores_i = self.predictor(zu_neg, zi_pos)

        scores = torch.cat([pos_scores, neg_scores_u, neg_scores_i], dim=-1)
        loss = -F.log_softmax(scores, 1)[:, 0].mean()
        return loss
