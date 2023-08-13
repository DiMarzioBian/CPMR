import torch
from torch import nn
from torch.nn import functional as F

from utils.misc import init_linear


class Expert(nn.Module):
    def __init__(self, d_input, d_latent):
        super().__init__()
        self.fc = nn.Linear(d_input, d_latent)
        init_linear(self.fc)

    def forward(self, x):
        return F.relu(self.fc(x))


class Gate(nn.Module):
    def __init__(self, d_input, n_expert):
        super().__init__()
        self.fc = nn.Linear(d_input, n_expert, bias=False)
        init_linear(self.fc)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)


class CGC_2_3(nn.Module):
    def __init__(self, d_input, d_latent):
        super().__init__()

        self.d_input = d_input
        self.d_latent = d_latent

        self.expert_shared = Expert(d_input, d_latent)
        self.expert_pred = Expert(d_input, d_latent)
        self.expert_his = Expert(d_input, d_latent)
        self.expert_ctx = Expert(d_input, d_latent)

        self.gate_pred = Gate(d_input, 2)
        self.gate_his = Gate(d_input, 2)
        self.gate_ctx = Gate(d_input, 2)

    def forward(self, x_his, x_ctx):
        x = torch.hstack((x_his, x_ctx))
        xe_shared = self.expert_shared(x).unsqueeze(1)
        xe_pred = self.expert_pred(x).unsqueeze(1)
        xe_his = self.expert_his(x).unsqueeze(1)
        xe_ctx = self.expert_ctx(x).unsqueeze(1)

        g_pred = self.gate_pred(x)
        g_his = self.gate_his(x)
        g_ctx = self.gate_ctx(x)

        res_pred = torch.einsum('ne,ned ->ned', g_pred, torch.cat((xe_shared, xe_pred), 1)).sum(dim=1)
        res_his = torch.einsum('ne,ned ->ned', g_his, torch.cat((xe_shared, xe_his), 1)).sum(dim=1)
        res_ctx = torch.einsum('ne,ned ->ned', g_ctx, torch.cat((xe_shared, xe_ctx), 1)).sum(dim=1)

        return res_pred, res_his, res_ctx
