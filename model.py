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
            **kwargs : Any
        ) -> None:
        super(SpeakerRecognition, self).__init__()
        lstm_block = CustomLSTM

        self.lstm = lstm_block(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )

    def _forward(self, x):
        # d-vector's dim : 64
        out, (h, c) = self.lstm(x) 
        out = out[:, -1, :]
        return out

    def forward(self, x) -> torch.Tensor:
        context = self._forward(x)
        return context