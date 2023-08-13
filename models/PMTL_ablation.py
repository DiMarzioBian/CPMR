import torch
from torch import nn

from models.PMTL import Gate, Expert


class CGC_1_2(nn.Module):
    """ for no_ctx ablation """
    def __init__(self, d_input, d_latent):
        super().__init__()

        self.d_input = d_input
        self.d_latent = d_latent

        self.expert_shared = Expert(self.d_input, self.d_latent)
        self.expert_pred = Expert(self.d_input, self.d_latent)
        self.expert_x = Expert(self.d_input, self.d_latent)

        self.gate_pred = Gate(self.d_input, 2)
        self.gate_x = Gate(self.d_input, 2)

    def forward(self, x):
        xe_shared = self.expert_shared(x).unsqueeze(1)
        xe_pred = self.expert_pred(x).unsqueeze(1)
        xe_x = self.expert_x(x).unsqueeze(1)

        g_pred = self.gate_pred(x)
        g_x = self.gate_x(x)

        res_pred = torch.einsum('ne,ned ->ned', g_pred, torch.cat((xe_shared, xe_pred), 1)).sum(dim=1)
        res_x = torch.einsum('ne,ned ->ned', g_x, torch.cat((xe_shared, xe_x), 1)).sum(dim=1)

        return res_pred, res_x
