import os
import torch
import torch.nn as nn

from torch.nn.modules.transformer import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.normalization import LayerNorm

from models.modules.transformer import PositionalEncoding


class PriorNet(nn.Module):
    def __init__(self, cfg, embed_dim):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = embed_dim
        self.cat_num = len(cfg.action_profiles)
        self.mean_actions = nn.Embedding(self.cat_num, embed_dim)
        self.logvar_actions = nn.Embedding(self.cat_num, embed_dim)
        self.input_size = (52*6+3) if cfg.input_rotation_format=="6dim" else (52*3+3)

        self.pre_coder1 = nn.Linear(self.input_size*cfg.S1_end_len, embed_dim)
        self.pre_coder2 = nn.Linear(self.input_size*cfg.S1_end_len, embed_dim)

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(embed_dim*cfg.S1_end_len)


    def forward(self, S1_end, labels):
        """
        Inputs:
            S1_end: last few frames of input query motion
            labels: input labels to generate
        Outputs:
            mu_tr, sigma_tr, mu_2, sigma_2
        """

        # encoded previous motion
        pre_coded_S1_end_mu_tr = self.pre_coder1(S1_end.reshape(-1, self.input_size*self.cfg.S1_end_len)).reshape(-1, self.embed_dim).unsqueeze(1)
        pre_coded_S1_end_logvar_tr = self.pre_coder2(S1_end.reshape(-1, self.input_size*self.cfg.S1_end_len)).reshape(-1, self.embed_dim).unsqueeze(1)

        # prevnet: labels[:, :1] holds the action label "transition" and encoded previous motion
        mu_tr = self.mean_actions(labels[:, :1]) + pre_coded_S1_end_mu_tr
        logvar_tr = self.logvar_actions(labels[:, :1]) + pre_coded_S1_end_logvar_tr

        # currnet: labels[:, 1:] holds the target action label
        mu_2 = self.mean_actions(labels[:, 1:])
        logvar_2 = self.logvar_actions(labels[:, 1:])

        return torch.cat((mu_tr, mu_2), dim=1), torch.cat((logvar_tr, logvar_2), dim=1)
