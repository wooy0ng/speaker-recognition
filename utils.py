import os
from typing import *

def get_file_names(PATH: str) -> List:
    result = []
    for file in os.listdir(PATH):
        result.append(os.path.join(PATH, file))
    return result

