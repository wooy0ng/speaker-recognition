from torch.utils.data import DataLoader, Dataset
from typing import *

import librosa
import torch
import torch.nn as nn
import utils
import numpy as np


def crop_padding(data: np.ndarray, sr: int, point: int) -> np.ndarray:
    if data.shape[0] < point:
        size = data.shape[0]
        pad = np.zeros(sr * (int(size/sr)+1) - size)
        data = np.append(data, pad)
    else:
        data = data[:point]
    return data


class LoadDataset(nn.Module):
    def __init__(self, 
            path: str,
            limit: int
        ) -> None:
        self.file_names =  utils.get_file_names(path)
        
        '''
        # feature extraction
        1. feature extraction using MFCC
        '''    
        result = []
        for file in self.file_names:
            data, sr = librosa.load(file)
            point = sr * limit
            data = crop_padding(data, sr, point)
            
            mfcc_data = librosa.feature.mfcc(
                y=data,
                sr=sr,
                n_mfcc=40,
            )
            result.append(mfcc_data)    # (20, 216)
        self.dataset = np.asarray(result)
        self.dataset = np.transpose(self.dataset, (0, 2, 1))
        return
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> torch.Tensor:
        return torch.FloatTensor(self.dataset[idx])

    def size(self, dim) -> int:
        return self.dataset.shape[dim]