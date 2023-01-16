import torch
import torch.nn as nn
import math
import numpy as np
import copy

from data.datautils.babel_label import label_over_twenty

from typing import Optional, Any
import torch.functional as F
from torch import Tensor
from torch.nn.modules.transformer import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.normalization import LayerNorm


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, device=None, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = torch.transpose(pe, 0, 1)
        if device is not None:
            pe = pe.to(device)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
            x shape = (batch * seq_len * embed_dim)
            pe.shape = (seq_len * max_len) -> slice into (seq_len * embed_dim)
        """
        x = x + self.pe[:, :x.shape[2]]
        return x


class LayeredPositionalEncoding(nn.Module):
    def __init__(self, d_model, device=None, max_len=5000):
        super(LayeredPositionalEncoding, self).__init__()
        pe_layer = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe_layer[:, 0::2] = torch.sin(position * div_term)
        pe_layer[:, 1::2] = torch.cos(position * div_term)
        pe_layer = torch.transpose(pe_layer, 0, 1)

        pe = torch.zeros((119, d_model, max_len))
        for i in range(119):
            pe[i] = pe_layer
            pe[i, i:, int(max_len/2):] = pe_layer[:d_model-i, int(max_len/2):]
        self.register_buffer("pe", pe)

    def forward(self, x, transition_len):
        pe = self.pe[transition_len]
        x = x + pe
        return x


class CVAETransformerEncoder(nn.Module):
    def __init__(
        self,
        cfg,
        embed_dim
    ):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = embed_dim

        if cfg.input_rotation_format == "6dim":
            input_dim = 6
        else:
            input_dim = 3

        # pre-coding
        self.pre_coding = nn.Linear(52 * input_dim + 3 + embed_dim, embed_dim)

        # learnable tokens
        self.action_embed = nn.Embedding(len(cfg.action_profiles), embed_dim)

        # transformer encoder layer
        spec = cfg.Transformer_spec
        encoder_layer = TransformerEncoderLayer(
            d_model=spec["embed_dim"],
            nhead=spec["nhead"],
            dim_feedforward=spec["dim_feedforward"],
            dropout=spec["dropout"],
            activation=spec["activation"],
        )
        encoder_norm = LayerNorm(
            spec["embed_dim"]
        )

        self.encoder = TransformerEncoder(encoder_layer, spec["enc_layers"], encoder_norm)

        # postprocessing layers for mean
        self.mean_conv = nn.Conv2d(1, spec["embed_dim"], (5, spec["embed_dim"]), stride=1)
        self.mean_fc = nn.Linear(spec["embed_dim"], spec["embed_dim"])

        # postprocessing layers for logvar
        self.logvar_conv = nn.Conv2d(1, spec["embed_dim"], (5, spec["embed_dim"]), stride=1)
        self.logvar_fc = nn.Linear(spec["embed_dim"], spec["embed_dim"])

        self.bn = nn.BatchNorm2d(spec["embed_dim"])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, encoder_input, label_mask, valid_mask=None, subaction_mean_mask=None):
        """
            input:
                encoder_input ( batch * duration * (smpl_param_num + 3) )
                label_mask ( batch * duration )
                valid_mask ( batch * duration ): batch input이 duration t에서 valid하면 1, dummy이면 0

            output mean, logvar of shape batch * embed_dim
            transformer encoder, decoder만 seq * batch * dim 순서
            나머지 과정에서는 batch * seq * dim 순서
        """
        batch_size, input_duration = encoder_input.shape[0:2]
        # encoder_input = encoder_input.reshape(batch_size*input_duration, 75)

        action_vec = self.action_embed(label_mask)
        encoder_input = torch.cat((encoder_input, action_vec), dim=2)
        pre_coded = self.pre_coding(encoder_input)

        # positional encoding
        pos_enc = PositionalEncoding(input_duration, label_mask.device).requires_grad_(False)
        encoder_input = pos_enc(pre_coded)

        # batch seq dim -> seq batch dim
        encoder_input = torch.transpose(encoder_input, 0, 1)

        # masking invalid part of the input data
        if valid_mask is not None:
            encoder_valid_mask = valid_mask == 0
        else:
            encoder_valid_mask = None
        
        # encoder
        encoded = self.encoder(encoder_input, src_key_padding_mask=encoder_valid_mask)  # encoder output (batch * (cfg.input_duration + 2), embed_dim)

        # seq batch dim -> batch seq dim
        encoded = torch.transpose(encoded, 0, 1)

        # pad to batch * (2 + seq + 2) * dim
        encoded = nn.functional.pad(encoded.unsqueeze(0), (0, 0, 2, 2), mode="replicate").squeeze(0).unsqueeze(1)

        # mean, logvar 1-d convolutional layers
        encoded_mean = self.mean_conv(encoded)
        encoded_mean = self.bn(encoded_mean)
        encoded_mean = self.relu(encoded_mean).transpose(1,3).squeeze(1)
        encoded_mean = self.mean_fc(encoded_mean)

        encoded_logvar = self.logvar_conv(encoded)
        encoded_logvar = self.bn(encoded_logvar)
        encoded_logvar = self.relu(encoded_logvar).transpose(1,3).squeeze(1)
        encoded_logvar = self.logvar_fc(encoded_logvar)

        mean = torch.matmul(torch.transpose(encoded_mean, 1, 2), subaction_mean_mask).transpose(1, 2)
        logvar = torch.matmul(torch.transpose(encoded_logvar, 1, 2), subaction_mean_mask).transpose(1, 2)

        return mean[:, 1:, :], logvar[:, 1:, :]


class CVAETransformerDecoder(nn.Module):
    def __init__(
        self,
        cfg,
        embed_dim
    ):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = embed_dim

        # transformer decoder layer
        spec = cfg.Transformer_spec 
        decoder_layer = TransformerDecoderLayer(
            d_model=spec["embed_dim"],
            nhead=spec["nhead"],
            dim_feedforward=spec["dim_feedforward"],
            dropout=spec["dropout"],
            activation=spec["activation"],
        )
        decoder_norm = LayerNorm(spec["embed_dim"])
        
        self.decoder = TransformerDecoder(decoder_layer, spec["dec_layers"], decoder_norm)
        batch_index = torch.tensor([[i]*cfg.max_input_len for i in range(60)]).reshape(-1)
        self.register_buffer("batch_index", batch_index, persistent=False)

        if self.cfg.layered_pos_enc:
            self.pos_enc = LayeredPositionalEncoding(cfg.max_input_len, max_len=embed_dim)
        else:
            self.pos_enc = PositionalEncoding(cfg.max_input_len)

        if cfg.output_rotation_format == "6dim":
            input_dim = 6
        else:
            input_dim = 3

        self.post_conv = nn.Conv2d(1, input_dim*52+3, (5, spec["embed_dim"]), stride=1)


    
    def forward(self, z, tgt_shape, output_mask=None, transition_len=None):
        batch_size, output_duration = tgt_shape[0:2]
        
        # process latent vectors
        zeroadd = torch.zeros((batch_size, 1, self.embed_dim), device=z.device)
        z_zeroadd = torch.cat((zeroadd, z), dim=1)
        batch_index = self.batch_index[:batch_size * output_duration]
        memory = z_zeroadd[batch_index, output_mask.reshape(-1), :].reshape(batch_size, output_duration, -1)

        # memory shape (batch dur+s1_end dim) -> (dur+s1_end batch dim)
        memory = memory.transpose(0, 1)

        target_mask = self.subsequent_mask(output_duration).to(z.device)  # mask size output_dur * output_dur
        target = torch.zeros((tgt_shape[0], tgt_shape[1], tgt_shape[2]), device=z.device, dtype=torch.float32)

        if self.cfg.layered_pos_enc:
            target = self.pos_enc(target, transition_len)
        else:
            target = self.pos_enc(target)

        # target shape batch seq dim -> seq batch dim
        target = torch.transpose(target, 0, 1)

        if output_mask is not None:
            decoder_valid_mask = output_mask == 0
        else:
            decoder_valid_mask = None

        decoded = self.decoder(
            tgt=target, 
            tgt_mask=target_mask, 
            memory=memory,
            tgt_key_padding_mask=decoder_valid_mask, 
            memory_key_padding_mask=decoder_valid_mask)

        # seq batch dim -> batch seq dim
        decoded = torch.transpose(decoded, 0, 1)

        # pad to batch * (2 + seq + 2) * dim
        decoded = nn.functional.pad(decoded.unsqueeze(0), (0, 0, 4, 0), mode="replicate").squeeze(0).unsqueeze(1)

        output = self.post_conv(decoded).transpose(1,3).squeeze(1)
        rot6d_out = output[:, :, :-3]
        trans_out = output[:, :, -3:]

        return rot6d_out, trans_out


    def subsequent_mask(self, size):
        atten_shape = (size, size)
        mask = np.triu(np.ones(atten_shape), k=1).astype('uint8') # masking with upper triangle matrix
        return torch.from_numpy(mask)== 1 # reverse (masking=True, non-masking=False)
