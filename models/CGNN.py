import math
import numpy as np
from scipy import special

import torch
from torch import nn


class InvNet(nn.Module):
    def __init__(self, order):
        super().__init__()
        self.order = order

    def forward(self, adj, x, alpha):
        z_stack = [x]
        z = x
        for _ in range(self.order):
            z = alpha * (adj @ z)
            z_stack.append(z)
        return torch.stack(z_stack, 0).sum(0)


class ExpNet(nn.Module):
    def __init__(self, order):
        super().__init__()
        self.order = order

        # compute bessel coefficients
        c_bessel = 2 * (special.jv(np.arange(order + 1), 0 - 1j) * (0 + 1j) ** np.arange(order + 1)).real
        c_bessel[0] /= 2
        self.register_buffer('c_bessel', torch.tensor(c_bessel, dtype=torch.float32).reshape(-1, 1, 1))

    def forward(self, adj, x, alpha):
        # Recursion of 1st kind Chebyshev polynomials
        pp_state = x
        p_state = alpha * (adj @ x)
        zs = [pp_state, p_state]
        for _ in range(self.order - 1):
            n_state = 2 * alpha * (adj @ p_state) - pp_state
            zs.append(n_state)
            pp_state, p_state = p_state, n_state
        return (torch.stack(zs, 0) * self.c_bessel).sum(0)


class CGNN(nn.Module):
    """ CGNN """
    def __init__(self, args):
        super().__init__()
        self.ts_max = args.ts_max
        self.d_latent = args.d_latent
        self.n_nodes = args.n_user + args.n_item

        self.exp_net = ExpNet(order=args.k_exp)
        self.inv_net = InvNet(order=args.k_inv)

        self.alpha = nn.Parameter(torch.ones(self.n_nodes) * 3)

    def forward(self, x_embed, x_t, t_step, adj):
        z = torch.cat([x_embed, x_t], 1) * math.exp(t_step)

        # matrix exponential
        alpha = torch.sigmoid(self.alpha).unsqueeze(1)
        if t_step > 0:
            z = self.exp_net(adj, z, alpha)

        # matrix inverse
        x_embed_exp, x_t_exp = torch.split(z, self.d_latent, 1)
        x_embed_inv = self.inv_net(adj, x_embed_exp - x_embed, alpha).neg()

        return x_embed_inv + x_t_exp
