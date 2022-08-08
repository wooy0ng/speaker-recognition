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
    ge2e_dataloader = preprocessing(args, 'train')
    infinite_iter = infinite_iterator(ge2e_dataloader)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dvector = DvectorUsingLSTM(
        input_size=40,
        hidden_size=256,
        num_layers=3,
        embedding_size=256
    ).to(device)
    criterion = GE2ELoss().to(device)
    optimizer = optim.SGD(list(dvector.parameters())+list(criterion.parameters()), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=1000,
        gamma=0.5
    )
    
    n_speakers = args.n_speakers
    n_utterances = args.n_utterances

    running_losses = []
    for step in count(start=1):
        start = time.time()
        batch = next(infinite_iter).to(device)
        output = dvector(batch)
        output = output.view(n_speakers, n_utterances, -1)
        loss = criterion(output)

        optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
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
        running_losses.append(loss.item())
        if step % 50 == 0:
            print("[{} training step / {:.3f}sec] avg loss : {:.3f}".format(
                step, 
                time.time()-start ,
                sum(running_losses) / len(running_losses))
            )
    return  

def validation(args) -> None:

    return