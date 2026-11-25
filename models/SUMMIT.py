import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from torch import Tensor
from copy import deepcopy
from typing import Tuple

"""
Paper link: https://arxiv.org/abs/2405.16557
Ported (unmodified logic) from code_submit_v2/models/{scalable_numerical_embedding.py, transformer_encoder_block.py,
tesne.py} -- SUMMIT's own implementation is called "TESNE" in that codebase.

Usage:

from models.SUMMIT import TESNE
model = TESNE(d_model=144, num_heads=1, ff_dim=144, seq_len=24, NUM_LEN=40, CAT_LEN=2, num_layers=1, output_size=2)
logits = model(x_num_idx, x_num, x_num_mask, x_cat_idx, x_cat, x_cat_mask)  # (B, output_size)

One intentional deviation from the original (which trains output_size=1 with sigmoid + BCE): forward() here
returns raw logits, with no sigmoid, so this framework's num_class-way CrossEntropy convention (see
models/itransformer_cls.py) also works with output_size=num_class; set output_size=1 and apply sigmoid
yourself outside if BCE is wanted instead.
"""


class SinusoidalEmbedding3d(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_seq_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(dim=1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(2)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SNE(nn.Module):
    """Scalable Numerical Embedding: value_embedding(idx) scaled by the value itself; idx/value are both
    zeroed by the mask first (independent_padding=True sends missing entries to the reserved padding_idx=0,
    whose embedding row is always zero, on top of the value-side zeroing)."""

    def __init__(self, d_model: int = 128, value_LEN: int = None, independent_padding: bool = True, max_norm: float = None):
        super().__init__()
        assert value_LEN is not None and value_LEN > 0
        self.value_LEN = value_LEN
        self.independent_padding = independent_padding
        if independent_padding:
            self.value_embedding = nn.Embedding(num_embeddings=value_LEN + 1, embedding_dim=d_model, padding_idx=0, max_norm=max_norm)
        else:
            self.value_embedding = nn.Embedding(num_embeddings=value_LEN, embedding_dim=d_model, max_norm=max_norm)

    def forward(self, x_idx, x, x_mask):
        embedding_idx = x_idx * x_mask if self.independent_padding else x_idx
        embedding_value = x * x_mask if self.independent_padding else x
        x_vec = self.value_embedding(embedding_idx)
        x_emb = embedding_value.unsqueeze(-1) * x_vec
        return x_emb


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, ff_dropout: float = 0.1, attn_dropout: float = 0.1,
                 norm_type: str = "LayerNorm", seq_len: int = None, num_value: int = None):
        super().__init__()
        assert seq_len is not None and num_value is not None
        self.seq_len, self.num_value, self.num_heads = seq_len, num_value, num_heads
        self.enc_self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=attn_dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(ff_dropout), nn.Linear(ff_dim, d_model))
        assert norm_type in ("LayerNorm", "BatchNorm1d")
        if norm_type == "LayerNorm":
            self.attn_norm, self.ff_norm = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        else:
            self.attn_norm = nn.BatchNorm1d(self.seq_len * self.num_value)
            self.ff_norm = nn.BatchNorm1d(self.seq_len * self.num_value)
        self.attn_dropout, self.ff_dropout = nn.Dropout(attn_dropout), nn.Dropout(ff_dropout)

    def forward(self, src: Tensor, src_mask: Tensor, src_key_padding_mask: Tensor):
        output, attn_weight = self.enc_self_attn(query=src, key=src, value=src, key_padding_mask=src_key_padding_mask, attn_mask=src_mask)
        output = src + self.attn_dropout(output)
        output = self.attn_norm(output)
        output_ = self.ff(output)
        output = output + self.ff_dropout(output_)
        output = self.ff_norm(output)
        return output, attn_weight


class Encoder(nn.Module):
    def __init__(self, encoder_layer: EncoderLayer, num_layers: int, only_mask_first: bool = False):
        super().__init__()
        self.seq_len, self.num_value, self.num_heads = encoder_layer.seq_len, encoder_layer.num_value, encoder_layer.num_heads
        self.only_mask_first = only_mask_first
        self.layers = nn.ModuleList([deepcopy(encoder_layer) for _ in range(num_layers)])

    def get_key_padding_mask(self, mask: Tensor) -> Tensor:
        flatten_mask = mask.reshape(mask.size(0), -1)
        return ~flatten_mask.bool()

    def forward(self, src: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        src_key_padding_mask = self.get_key_padding_mask(mask)
        attns_all = []
        output = src
        counter = 0 if self.only_mask_first else -1
        for layer in self.layers:
            if counter == -1:
                output, attn = layer(src=output, src_mask=None, src_key_padding_mask=src_key_padding_mask)
                attns_all.append(attn)
            elif counter == 0:
                output, attn = layer(src=output, src_mask=None, src_key_padding_mask=src_key_padding_mask)
                attns_all.append(attn)
                counter += 1
            else:
                output, attn = layer(src=output, src_mask=None, src_key_padding_mask=None)
                attns_all.append(attn)
        return output, torch.stack(attns_all)


class TESNE(nn.Module):
    def __init__(self, d_model: int = 128, num_heads: int = 2, ff_dim: int = 1024, ff_dropout: float = 0.05,
                 attn_dropout: float = 0.05, norm_type: str = "LayerNorm", only_mask_first: bool = False,
                 seq_len: int = None, NUM_LEN: int = None, CAT_LEN: int = None, num_layers: int = 5,
                 decoder_down_factor: int = 4, decoder_dropout: float = 0.5, output_size: int = 1,
                 num_embedding_module_kwargs: dict = {}, cat_embedding_module_kwargs: dict = {}):
        super().__init__()
        assert seq_len is not None and seq_len > 0
        assert NUM_LEN is not None and NUM_LEN > 0
        assert CAT_LEN is not None and CAT_LEN > 0
        self.seq_len, self.d_model = seq_len, d_model

        self.num_embedding = SNE(d_model=d_model, value_LEN=NUM_LEN, **num_embedding_module_kwargs)
        self.cat_embedding = SNE(d_model=d_model, value_LEN=CAT_LEN, **cat_embedding_module_kwargs)
        self.pos_emb = SinusoidalEmbedding3d(d_model, dropout=0.1, max_seq_len=5000)

        encoder_layer = EncoderLayer(d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, ff_dropout=ff_dropout,
                                      attn_dropout=attn_dropout, norm_type=norm_type, seq_len=seq_len, num_value=NUM_LEN + CAT_LEN)
        self.encoder = Encoder(encoder_layer=encoder_layer, num_layers=num_layers, only_mask_first=only_mask_first)

        self.decoder = nn.Sequential(nn.Linear(d_model, d_model // decoder_down_factor), nn.GELU(),
                                      nn.Dropout(decoder_dropout), nn.Linear(d_model // decoder_down_factor, output_size))

    def forward(self, x_num_idx, x_num, x_num_mask, x_cat_idx, x_cat, x_cat_mask):
        x_num_embedded = self.num_embedding(x_num_idx, x_num, x_num_mask)
        x_cat_embedded = self.cat_embedding(x_cat_idx, x_cat, x_cat_mask)
        x = torch.cat([x_num_embedded, x_cat_embedded], dim=2)
        x_mask = torch.cat([x_num_mask, x_cat_mask], dim=2)
        x = self.pos_emb(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x_embedded, _ = self.encoder(x, x_mask)
        last_vec = torch.mean(x_embedded, dim=1)
        output = self.decoder(last_vec)
        return output
