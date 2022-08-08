from LoadDataset import *
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def preprocessing(args, mode: str):
    '''### Return GE2E DataLoader'''
    if mode == 'train':
        path = args.train_path
    else:
        path = args.val_path
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
    ge2e_dataloader = DataLoader(GE2E_dataset, batch_size=args.n_speakers)
    ge2e_dataloader = collate_batch(ge2e_dataloader)
    return ge2e_dataloader

def collate_batch(batch):
    """
    Collate a whole batch of utterances.
    return size : (n_utterances, batch, seg_len, n_mels)
    """
    flatten = [u for s in batch for u in s]
    return pad_sequence(flatten, batch_first=True, padding_value=0)