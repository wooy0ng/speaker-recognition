import numpy as np
import pandas as pd
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLSTM(nn.Module):
    def __init__(self, 
            input_size : int,
            hidden_size : int,
            num_layers : int,
        ) -> None:
        super(CustomLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lstm(x)
        return x

class SpeakerRecognition(nn.Module):
    def __init__(self,
            input_size : int,
            hidden_size : int,
            num_layers : int,
            batch_size: int,
            **kwargs : Any
        ) -> None:
        super(SpeakerRecognition, self).__init__()
        lstm_block = CustomLSTM
        self.kwargs = kwargs
        self.batch_size = batch_size
        self.contexts = []

        self.lstm = lstm_block(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        self.clf = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*4, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size*4 ,hidden_size, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1, bias=True),
            nn.Sigmoid()
        )
        self.init_weights()

    def init_weights(self) -> None:
        for name, param in self.named_parameters():
            nn.init.normal_(param)
        return

    def _forward(self, x) -> Optional[torch.Tensor]:
        # d-vector
        d_vectors, (h, c) = self.lstm(x)
        if 'normalize' in self.kwargs.keys():
            if self.kwargs['normalize'] is True:
                d_vectors = F.normalize(d_vectors, p=2, dim=2)
        # d_vectors_avg = torch.mean(d_vectors, dim=1)
        # clf = self.clf(d_vectors_avg)
        self.context = d_vectors[:, -1, :]
        self.contexts.append(self.context)
        if len(self.contexts) < self.batch_size:
            return
        contexts = torch.stack(self.contexts)
        self.contexts = []
        if self.batch_size > 1:
            out = contexts.mean(dim=0)
        else:
            out = contexts.squeeze(0)
        return out

    def forward(self, x) -> Optional[torch.Tensor]:
        out = self._forward(x)
        return out