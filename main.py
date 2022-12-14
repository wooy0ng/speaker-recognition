import argparse
import mode

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', required=False, default='train',
        help='--mode [preprocess | train | train_after | validation | visualization]'
    )
        
    # ./model/train_dataset
    # /mnt/e/project/ver1.1/model/train_dataset
    parser.add_argument('--preprocessing_path', required=False, default='./model/train_dataset',
        help='--preprocessing_path [train_dataset | test_dataset]'
    )   

    parser.add_argument('--train_path', required=False, default='../voxceleb_dataset/train')
    parser.add_argument('--val_path', required=False, default='../voxceleb_dataset/test')

    parser.add_argument('--model_path', required=False, default='./model')
    parser.add_argument('--is_preprocessed', required=False, default=True)

    parser.add_argument('--n_utterances', required=False, default=10)
    parser.add_argument('--min_segment', required=False, default=160)
    parser.add_argument('--n_speakers', required=False, default=64)
    parser.add_argument('--train_test_split', required=False, default=False)
    
    args = parser.parse_args()
    if args.mode == 'preprocess':
        mode.preprocess(args)
    elif args.mode == 'train':
        mode.train(args)
    elif args.mode == 'train_after':
        mode.train_after(args)
    elif args.mode == 'validation':
        mode.validation(args)
    elif args.mode == 'visualization':
        mode.visualization(args)