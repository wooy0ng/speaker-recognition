from itertools import count
import numpy as np
import pandas as pd
from LoadDataset import *
from preprocessing import preprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl
import random
from torch.utils.data import DataLoader
from ge2eloss import GE2ELoss

from model import DvectorUsingLSTM
from model_after import *
import time

from utils import *

def infinite_iterator(dataloader):
    """Infinitely yield a batch of data."""
    while True:
        for batch in iter(dataloader):
            yield batch

def train(args) -> None:
    preprocessing_path = Path(args.preprocessing_path)
    preprocessing_path.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = preprocessing(args, 'train', split=args.train_test_split)
    
    train_iter = infinite_iterator(train_loader)
    val_iter = None
    if val_loader is not None:
        val_iter = infinite_iterator(val_loader)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_path = Path(args.model_path)

    dvector = DvectorUsingLSTM(
        input_size=40,
        hidden_size=256,
        num_layers=3,
        embedding_size=256,
        seg_len=160
    ).to(device)
    dvector = torch.jit.script(dvector)

    criterion = GE2ELoss().to(device)
    optimizer = optim.SGD(list(dvector.parameters())+list(criterion.parameters()), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=100000,
        gamma=0.5
    )
    
    n_speakers = args.n_speakers
    n_utterances = args.n_utterances

    train_running_loss = []
    val_running_loss = []
    for step in count(start=1):
        start = time.time()
        batch = next(train_iter).to(device)
        output = dvector(batch)
        output = output.view(n_speakers, n_utterances, -1)
        loss = criterion(output)

        optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(dvector.parameters())+list(criterion.parameters()),
            max_norm=3,
            norm_type=2
        )
        dvector.lstm.embedding.weight.grad *= 0.5
        dvector.lstm.embedding.bias.grad *= 0.5
        criterion.w.grad *= 0.01
        criterion.b.grad *= 0.01

        optimizer.step()
        scheduler.step()
        train_running_loss.append(loss.item())
        if step % 50 == 0:
            if val_iter is not None:
                for _ in range(50):
                    batch = next(val_iter).to(device)
                    with torch.no_grad():
                        output = dvector(batch).view(n_speakers, n_utterances, -1)
                        loss = criterion(output)
                        val_running_loss.append(loss.item())
            print("\n----------")
            print("({} training step / {:.3f}sec)".format(step, time.time()-start))
            print("train avg loss : {:.3f}".format(
                sum(train_running_loss) / len(train_running_loss))
            )
            try:
                print("validation avg loss : {:.3f}".format(
                    sum(val_running_loss) / len(val_running_loss))
                )
            except ZeroDivisionError:
                print("validation avg loss : -")
        if step % 10000 == 0:
            save_path = checkpoint_path / f'model-step{step}.pt'
            dvector.cpu()
            dvector.save(str(save_path))
            dvector.to(device)
    return  

def train_after(args) -> None:
    dvector = torch.jit.load('./model/model-step250000.pt')

    wav, sample_rate = torchaudio.load('../voxceleb_dataset/score/id10022_00002.wav')
    wav2mel = Wav2Mel()
    mel_wav = wav2mel(wav, sample_rate)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    key_size = 128
    memorized_model = MemorizedModel(256, key_size).to(device)
    memorized_model = torch.jit.script(memorized_model)

    optimizer = optim.Adam(memorized_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    epochs = 5
    for epoch in range(epochs):
        embeded_vector = dvector.embed_utterance(mel_wav).to(device)
        key = make_random_key(key_size).to(device)
        
        output = memorized_model(embeded_vector)
        loss = criterion(key, output)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"{epoch+1} epoch loss : {loss.item():.3f}")
    memorized_model.cpu()
    memorized_model.save('./memorized_model.pt')
    memorized_model.to(device)

    return

def validation(args) -> None:
    dvector = torch.jit.load('./model/model-step250000.pt')

    wav1, sample_rate1 = torchaudio.load('../voxceleb_dataset/score/id10022_00002.wav') # true
    wav2, sample_rate2 = torchaudio.load('../voxceleb_dataset/score/id10004_00001.wav') # false
    wav2mel = Wav2Mel()
    
    mel_wav1 = wav2mel(wav1, sample_rate1)
    mel_wav2 = wav2mel(wav2, sample_rate2)

    key_size = 128
    
    key = make_random_key(key_size)
    dvector = torch.jit.load('./model/model-step250000.pt')
    memorized_model = torch.jit.load('./memorized_model.pt')
    
    with torch.no_grad():
        embed_vector1 = dvector.embed_utterance(mel_wav1)
        embed_vector2 = dvector.embed_utterance(mel_wav2)
        
        # print(embed_vector1)
        # print(embed_vector2)
        output1 = memorized_model(embed_vector1)
        output2 = memorized_model(embed_vector2)

        t1 = get_variance(key, output1)
        t2 = get_variance(key, output2)
        print(bit_to_hex(key))
        print(bit_to_hex(torch.where(output1 > 0.5, 1, 0)))
        print(bit_to_hex(torch.where(output2 > 0.5, 1, 0)))
    return

def get_variance(a, b):
    return torch.mean((a - b) ** 2, dim=0)
