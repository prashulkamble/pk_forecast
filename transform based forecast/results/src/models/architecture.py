import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import LayerNorm
from torch.nn.init import xavier_uniform_
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import numpy as np


class TransformerModel(nn.Module):
    def __init__(
        self,
        src_seq_len,
        d_model,
        nhead,
        nhid,
        nlayers,
        dropout=0.5,
        activation="relu",
        cat_embs=None,
        fc_dims=[],
        use_src_mask=False,
        use_memory_mask=False,
        device=None,
    ):
        super(TransformerModel, self).__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if cat_embs is not None:
            self.cat_embs = [
                nn.Embedding(n_items, emb_size)
                if emb_size != 0
                else nn.Embedding(n_items, n_items)
                for n_items, emb_size in cat_embs.values()
            ]
            for i, (n_items, emb_size) in enumerate(cat_embs.values()):
                if emb_size == 0:
                    self.cat_embs[i].weight.data = torch.eye(
                        n_items, requires_grad=False
                    )
                    for param in self.cat_embs[i].parameters():
                        param.requires_grad = False

            total_cat_emb_size = np.array(
                [
                    emb_size if emb_size != 0 else n_items
                    for n_items, emb_size in cat_embs.values()
                ]
            ).sum()
        else:
            self.cat_embs = None
            total_cat_emb_size = 0

        self.tgt_mask = None
        self.src_mask = None
        self.memory_mask = None
        self.use_src_mask = use_src_mask
        self.use_memory_mask = use_memory_mask
        self.pos_encoder = PositionalEncoding(d_model + total_cat_emb_size, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model + total_cat_emb_size, nhead, nhid, dropout, activation
        )
        encoder_norm = LayerNorm(d_model + total_cat_emb_size)
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, nlayers, encoder_norm
        )

        decoder_layers = TransformerDecoderLayer(
            d_model + total_cat_emb_size, nhead, nhid, dropout, activation
        )
        decoder_norm = LayerNorm(d_model + total_cat_emb_size)
        self.transformer_decoder = TransformerDecoder(
            decoder_layers, nlayers, decoder_norm
        )

        if len(fc_dims) > 0:
            fc_layers = []
            for i, hdim in enumerate(fc_dims):
                if i != 0:
                    fc_layers.append(nn.Linear(fc_dims[i - 1], hdim))
                    fc_layers.append(nn.Dropout(dropout))
                else:
                    fc_layers.append(nn.Linear(d_model + total_cat_emb_size, hdim))
                    fc_layers.append(nn.Dropout(dropout))

            self.fc = nn.Sequential(*fc_layers)
            self.output = nn.Linear(fc_dims[-1], 1)
        else:
            self.fc = None
            self.output = nn.Linear(d_model + total_cat_emb_size, 1)

        self._reset_parameters()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(
        self, src=None, src_cat_idx=None, tgt=None, tgt_cat_idx=None, memory=None
    ):
        if src is not None:
            src = Variable(src, requires_grad=True).to(self.device).float()

            if src_cat_idx is not None:
                src_cat = torch.cat(
                    [
                        emb_layer(src_cat_idx[:, :, cat_i])
                        for cat_i, emb_layer in enumerate(self.cat_embs)
                    ],
                    dim=-1,
                )
                src = torch.cat([src_cat.to(self.device), src], dim=-1)

            src = self.pos_encoder(src)

            if self.use_src_mask:
                if self.src_mask is None or self.src_mask.size(0) != len(src):
                    mask = self._generate_square_subsequent_mask(len(src)).to(
                        self.device
                    )
                    self.src_mask = mask

            memory = self.transformer_encoder(src, mask=self.src_mask)

        if tgt is None:
            return memory
        else:
            tgt = Variable(tgt, requires_grad=True).to(self.device).float()

            if tgt_cat_idx is not None:
                tgt_cat = torch.cat(
                    [
                        emb_layer(tgt_cat_idx[:, :, cat_i])
                        for cat_i, emb_layer in enumerate(self.cat_embs)
                    ],
                    dim=-1,
                )
                tgt = torch.cat([tgt_cat.to(self.device), tgt], dim=-1)

            #             tgt = self.pos_encoder(tgt)

            if self.tgt_mask is None or self.tgt_mask.size(0) != len(tgt):
                mask = self._generate_square_subsequent_mask(len(tgt)).to(self.device)
                self.tgt_mask = mask

            if self.use_memory_mask:
                if self.memory_mask is None or self.memory_mask.size(0) != len(memory):
                    mask = self._generate_square_subsequent_mask(len(memory)).to(
                        self.device
                    )
                    self.memory_mask = mask

            decoder_output = self.transformer_decoder(
                tgt, memory, memory_mask=self.memory_mask, tgt_mask=self.tgt_mask
            )

            fc_input = decoder_output

            if self.fc is not None:
                fc_output = self.fc(fc_input)
            else:
                fc_output = fc_input

            output = self.output(fc_output)
            #             output = F.relu(output)
            return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

