import argparse
import mode

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', required=False, default='train',
        help='--mode [train | validation]'
    )
    parser.add_argument('--train_path', required=False, default='../voxceleb_dataset/dev')
    parser.add_argument('--val_path', required=False, default='../voxceleb_dataset/test')
    parser.add_argument('--preprocessing_path', required=False, default='./model/dataset')
    parser.add_argument('--model_path', required=False, default='./model')
    parser.add_argument('--is_preprocessed', required=False, default=True)
    parser.add_argument('--n_utterances', required=False, default=10)
    parser.add_argument('--min_segment', required=False, default=160)
    parser.add_argument('--n_speakers', required=False, default=64)
    parser.add_argument('--train_test_split', required=False, default=False)
    
    args = parser.parse_args()

    if args.mode == 'train':
        mode.train(args)
    elif args.mode == 'validation':
        mode.validation(args)
    