import numpy as np
import pandas as pd
from LoadDataset import LoadDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl
from torch.utils.data import DataLoader

from model import SpeakerRecognition

def train(args) -> None:
    if args.mode == 'train':
        path = args.train_path
    dataset = LoadDataset(path, args.limit)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpeakerRecognition(
        input_size=dataset.size(2),
        hidden_size=64,
        num_layers=3
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    pdist = nn.PairwiseDistance(p=2)
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    
    epochs = 20
    label = {
        'accept': torch.FloatTensor([1]), 
        'reject': torch.FloatTensor([0])
    }
    threshold = None
    count = 0
    for epoch in range(epochs):
        result = []
        for data in train_loader:
            data = data.to(device)
            result.append(context:=model(data))
            if len(result) == len(train_loader):
                result = torch.stack(result).squeeze()
                contexts_avg = torch.mean(result, dim=0).unsqueeze(dim=0)

                loss = 1.0 - F.cosine_similarity(contexts_avg, context)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if threshold is None:
            threshold = loss.item()
        else:
            if ((threshold < loss.item() or int(threshold) == 0) and count > 3):
                print("{} / {} epoch\n----------".format(epoch+1, epochs))
                print(f"running loss : {loss.item():f}")    
                print("break"); break
            threshold = loss.item()
            count += 1
        print("{} / {} epoch\n----------".format(epoch+1, epochs))
        print(f"running loss : {loss.item():.3f}\n")

    # object save
    pkl.dump(contexts_avg, open(f'contexts_avg.pkl', 'wb+'))
    torch.save(model.state_dict(), './model/pretrain.pt')
    return  

def validation(args) -> None:
    if args.mode == 'validation':
        path = args.val_path
    dataset = LoadDataset(path, args.limit)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpeakerRecognition(
        input_size=dataset.size(2),
        hidden_size=64,
        num_layers=3
    ).to(device)

    contexts_avg = pkl.load(open('contexts_avg.pkl', 'rb+'))
    model.load_state_dict(torch.load('./model/pretrain.pt'))
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            data = data.to(device)
            context = model(data)
        
            cosine_sim = F.cosine_similarity(contexts_avg, context)
            print(dataset.file_names[idx] + ' : {:.4f}'.format(cosine_sim.item()))
    return