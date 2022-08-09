from LoadDataset import *
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

def preprocessing(args, mode: str, split: bool):
    '''### Return GE2E DataLoader'''
    if mode == 'train':
        path = args.train_path
    elif mode == 'validation':
        path = args.val_path
    else:
        path = args.score_path
    mel_dataset = MelDataset(path)
    mel_dataloader = DataLoader(mel_dataset, batch_size=1)

    speakers_info = {speaker: [] for speaker in mel_dataset.speakers}
    for speaker, mel_wav in mel_dataloader:
        speaker = speaker[0]
        mel_wav = mel_wav.squeeze(0)
        speakers_info[speaker].append(
            {
                'mel_tensor': mel_wav,
                'seg_len': len(mel_wav)
            }
        )
    GE2E_dataset = GE2EDataset(
        speakers_info=speakers_info, 
        n_utterances=args.n_utterances, 
        min_segment=args.min_segment,
    )
    if split is True:
        train_set, valid_set = train_test_split(GE2E_dataset, test_size=0.5, random_state=42)
    else:
        train_set = GE2E_dataset
        valid_set = None
    train_dataloader = collate_batch(DataLoader(train_set, batch_size=args.n_speakers))
    val_dataloader = None

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