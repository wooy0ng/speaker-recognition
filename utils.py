import os
from typing import *
from pathlib import Path

def get_speakers(PATH: str) -> List:
    result = dict()
    for speaker in os.listdir(PATH):
        infos = []
        for utterances in os.listdir(os.path.join(PATH, speaker)):
            files = os.listdir(utterance:=os.path.join(PATH, speaker, utterances))
            infos.append([os.path.join(utterance, file) for file in files])
        result[speaker] = infos
    return result

