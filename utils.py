import os
from typing import *
from pathlib import Path
import torch

def get_speakers(PATH: str) -> List:
    result = dict()
    for speaker in os.listdir(PATH):
        infos = []
        for utterances in os.listdir(os.path.join(PATH, speaker)):
            files = os.listdir(utterance:=os.path.join(PATH, speaker, utterances))
            infos.append([os.path.join(utterance, file) for file in files])
        result[speaker] = infos
    return result

def bit_to_string(data):
    data = data.to(dtype=torch.int8)
    data = str(data.tolist()).lstrip('[').rstrip(']').split(', ')
    data = ''.join(data)
    return data

def string_to_bit(data):
    return torch.tensor([int(bit) for bit in data])

def bit_to_hex(data):
    return '%08X' % int(bit_to_string(data), 2)    

def make_random_key(key_size: int):
    if os.path.exists('./key.txt'):
        with open('./key.txt', 'r+') as f:
            random_key = string_to_bit(f.readline())
    else:
        random_key = torch.randint(0, 2, (key_size, ))
        with open('./key.txt', 'w+') as f:
            f.write(bit_to_string(random_key))
    return random_key.to(dtype=torch.float32)