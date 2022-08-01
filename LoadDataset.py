from more_itertools import sample
from torch.utils.data import DataLoader, Dataset
from typing import *
import librosa
import torchaudio
from torchaudio.sox_effects import apply_effects_tensor
from torchaudio.transforms import MelSpectrogram

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

class Wav2Mel(nn.Module):
    '''
    # Melspectrogram parameters
    - sr : sample rate of audio signal
    - n_mels : Number of mel filterbanks
    - n_fft : Size of FFT
    - hop_length : Length of hop between STFT windows
    - f_min : Minimum frequency
    '''
    def __init__(self,
            fft_window_ms: float,
            fft_hop_ms: float,
            f_min: float,
            n_mels: int,
            sr: int=22050,
        ) -> None:
        super(Wav2Mel, self).__init__()

        self.melspectrogram = MelSpectrogram(
            sample_rate=sr,
            n_fft=int(sr * fft_window_ms / 1000),
            hop_length=int(sr * fft_hop_ms / 1000),
            f_min=f_min,
            n_mels=n_mels,
        )

    def forward(self, wav: np.ndarray) -> torch.Tensor:
        # wav = torch.tensor(librosa.util.normalize(wav), dtype=torch.float32)
        wav = torch.tensor(wav, dtype=torch.float32)
        mel_data = self.melspectrogram(wav)
        mel_data = torch.log(torch.clamp(mel_data, min=1e-9))
        return mel_data

class LoadDataset(nn.Module):
    '''
    ### feature extraction
    feature extraction using librosa
    '''    
    def __init__(self, 
            path: str,
            limit: int,
            sr: int=22050,
        ) -> None:
        self.file_names = utils.get_file_names(path)
        wav2mel = Wav2Mel(
            sr=sr,
            fft_window_ms=25.,
            fft_hop_ms=10.,
            f_min=50.,
            n_mels=40
        )
        result = []
        for file in self.file_names:
            data, sr = librosa.load(file,
                sr=sr
            )
            point = sr * limit
            data = crop_padding(data, sr, point)
            mfcc_data = wav2mel(data)   # (feature, sequence)
            result.append(mfcc_data)
        self.dataset = torch.stack(result)  # (batch, feature, sequence)
        self.dataset = torch.transpose(self.dataset, 1, 2)
        return
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> torch.Tensor:
        return self.dataset[idx]

    def size(self, dim) -> int:
        return self.dataset.shape[dim]