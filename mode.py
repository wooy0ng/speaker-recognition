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
import time

def infinite_iterator(dataloader):
    """Infinitely yield a batch of data."""
    while True:
        for batch in iter(dataloader):
            yield batch

def train(args) -> None:
    train_loader, val_loader = preprocessing(args, 'train', split=args.train_test_split)
    train_iter = infinite_iterator(train_loader)
    val_iter = None
    if val_loader is not None:
        val_iter = infinite_iterator(val_loader)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dvector = DvectorUsingLSTM(
        input_size=40,
        hidden_size=256,
        num_layers=3,
        embedding_size=256,
        seg_len=160
    ).to(device)
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
    return  

def validation(args) -> None:
    
    return

def score(args) -> None:
    '''calculate equal error rate'''
    

    return