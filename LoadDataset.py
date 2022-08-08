from more_itertools import sample
from torch.utils.data import DataLoader, Dataset
from typing import *

import librosa
import torchaudio
from torchaudio.sox_effects import apply_effects_tensor
from torchaudio.transforms import MelSpectrogram
from pathlib import Path
import torch
import torch.nn as nn
import os
import numpy as np
import random


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
            sr: int=16000,
            fft_window_ms: float=25.0,
            fft_hop_ms: float=10.0,
            f_min: float=50.0,
            n_mels: int=40,
        ) -> None:
        super(Wav2Mel, self).__init__()

        self.melspectrogram = MelSpectrogram(
            sample_rate=sr,
            n_fft=int(sr * fft_window_ms / 1000),
            hop_length=int(sr * fft_hop_ms / 1000),
            f_min=f_min,
            n_mels=n_mels,
        )

    def forward(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        # wav = torch.tensor(librosa.util.normalize(wav), dtype=torch.float32)
        mel_data = self.melspectrogram(wav).squeeze(0).T
        mel_data = torch.log(torch.clamp(mel_data, min=1e-9))
        return mel_data

class MelDataset(Dataset):
    '''
    ### feature extraction
    feature extraction using librosa
    '''    
    def __init__(self, 
            path: str,
        ) -> None:
        
        # setup mel function
        wav2mel = Wav2Mel()

        # preprocessing
        self.speakers = set()
        self.infos = []
        
        speaker_paths = [x for x in Path(path).iterdir() if x.is_dir()]
        for speaker_path in speaker_paths:
            audio_paths = librosa.util.find_files(speaker_path)
            speaker_name = speaker_path.name
            self.speakers.add(speaker_name)
            for audio_path in audio_paths:
                # wav2mel
                wav, sr = torchaudio.load(audio_path)
                mel_wav = wav2mel(wav, sr)
                self.infos.append((speaker_name, mel_wav))
        return
        
    def __len__(self):
        return len(self.infos)

    def __getitem__(self, idx):
        return self.infos[idx]

class GE2EDataset(Dataset):
    def __init__(self,
        speakers_info: dict,
        n_utterances: int,
        min_segment: int,
    ) -> None:
        self.min_segment = min_segment
        self.n_utterances = n_utterances
        self.infos = []

        for uttrs_info in speakers_info.values():
            infos = [
                uttr_info['mel_tensor']
                for uttr_info in uttrs_info
                if uttr_info['seg_len'] > min_segment
            ]
            if len(infos) > n_utterances:
                self.infos.append(infos)
        return
    
    def __len__(self) -> int:
        return len(self.infos)

    def __getitem__(self, idx) -> List:
        uttrs = random.sample(self.infos[idx], self.n_utterances)
        lefts = [random.randint(0, len(uttr) - self.min_segment) for uttr in uttrs]
        segments = [
            uttr[left:left+self.min_segment, :] for uttr, left in zip(uttrs, lefts)
        ]
        return segments