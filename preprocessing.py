from LoadDataset import *
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from uuid import uuid4
import json

from tqdm import tqdm

def preprocessing(args, mode: str, split: bool):
    ''' Return GE2E DataLoader '''
    if mode == 'preprocess':
        path = args.train_path
    elif mode == 'train':
        path = args.train_path
    elif mode == 'validation':
        path = args.val_path
    else:
        path = args.score_path
    
    preprocessing_path = Path(args.preprocessing_path)
    if args.is_preprocessed is False:
        mel_dataset = MelDataset(path)
        mel_dataloader = DataLoader(mel_dataset, batch_size=1)

        speakers_info = {speaker: [] for speaker in mel_dataset.speakers}
        for speaker, mel_wav in tqdm(mel_dataloader):
            speaker = speaker[0]
            mel_wav = mel_wav.squeeze(0)
            random_path = preprocessing_path / f'uttrance-{uuid4().hex}.pt'
            torch.save(mel_wav, random_path)
            speakers_info[speaker].append(
                {
                    'feature_path': random_path.name,
                    'seg_len': len(mel_wav)
                }
            )
        with open(preprocessing_path / "metadata.json", 'w') as f:
            json.dump(speakers_info, f, indent=2)
    try:
        with open(preprocessing_path / "metadata.json", 'r') as f:
            speakers_info = json.load(f)
    except BaseException as e:
        with open(Path(path) / "metadata.json", 'r') as f:
            speakers_info = json.load(f)

    GE2E_dataset = GE2EDataset(
        preprocessing_path=preprocessing_path,
        speakers_info=speakers_info, 
        n_utterances=args.n_utterances, 
        min_segment=args.min_segment,
    )
    if split is True:
        train_set, valid_set = train_test_split(GE2E_dataset, test_size=0.5, random_state=42)
    else:
        train_set = GE2E_dataset
        valid_set = None
    print("[+] collate batch...", end=' ')
    train_dataloader = collate_batch(DataLoader(train_set, batch_size=args.n_speakers))
    val_dataloader = None
    print("complete")
    if valid_set is not None:
        val_dataloader = collate_batch(DataLoader(valid_set, batch_size=args.n_speakers))
    return train_dataloader, val_dataloader

def collate_batch(batch):
    """
    Collate a whole batch of utterances.
    return size : (n_utterances, batch, seg_len, n_mels)
    """
    flatten = [u for s in batch for u in s]
    return pad_sequence(flatten, batch_first=True, padding_value=0)