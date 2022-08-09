import numpy as np
import pandas as pd
import math
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicLSTM(nn.Module):
    def __init__(self, 
            input_size : int,
            hidden_size : int,
            num_layers : int,
            embedding_size: int,
        ) -> None:
        super(BasicLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.embedding = nn.Linear(hidden_size, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        embed = self.embedding(lstm_out[:, -1, :])
        out = embed.div(embed.norm(p=2, dim=-1, keepdim=True))
        return out

class AttentivePooledLSTMDvector(nn.Module):
    """
    ### attention pooling
    - LSTM-based d-vector with attentive pooling.
    """
    def __init__(
        self,
        num_layers=3,
        dim_input=40,
        dim_cell=256,
        dim_emb=256,
        seg_len=160,
    ):
        super().__init__()
        self.lstm = nn.LSTM(dim_input, dim_cell, num_layers, batch_first=True)
        self.embedding = nn.Linear(dim_cell, dim_emb)
        self.linear = nn.Linear(dim_emb, 1)
        self.seg_len = seg_len

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward a batch through network."""
        lstm_outs, _ = self.lstm(inputs)  # (batch, seg_len, dim_cell)
        embeds = torch.tanh(self.embedding(lstm_outs))  # (batch, seg_len, dim_emb)
        attn_weights = F.softmax(self.linear(embeds), dim=1)
        embeds = torch.sum(embeds * attn_weights, dim=1)
        return embeds.div(embeds.norm(p=2, dim=-1, keepdim=True))

class DvectorUsingLSTM(nn.Module):
    def __init__(self,
            input_size : int,
            hidden_size : int,
            num_layers : int,
            embedding_size: int,
            seg_len: int,
        ) -> None:
        super(DvectorUsingLSTM, self).__init__()
        self.seg_len = seg_len

        # lstm_block = BasicLSTM
        lstm_block = AttentivePooledLSTMDvector
        self.contexts = []

        if lstm_block.__name__ == 'BasicLSTM':
            self.lstm = lstm_block(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                embedding_size=embedding_size
            )
        else:
            self.lstm = lstm_block()

    def _forward(self, x) -> Optional[torch.Tensor]:
        # d-vector
        dvector = self.lstm(x)
        return dvector

    def forward(self, x):
        out = self._forward(x)
        return out

    def embed_utterance(self, utterance: torch.Tensor) -> torch.Tensor:
        if utterance.size(0) <= self.seg_len:
            embed = self.forward(utterance.unsqueeze(0)).squeeze(0)
        else:
            hop_len = self.seg_len // 2
            tgt_len = math.ceil(utterance.size(0) / hop_len) * hop_len
            zero_padding = torch.zeros(tgt_len - utterance.size(0), utterance.size(1))
            padded = torch.cat([utterance, zero_padding.to(utterance.device)])

            segments = padded.unfold(0, self.seg_len, self.seg_len // 2)
            segments = segments.transpose(1, 2)
            embeds = self.forward(segments)
            embed = embeds.mean(dim=0)
            embed = embed.div(embed.norm(p=2, dim=-1, keepdim=True))
        return embed
