import argparse
import mode

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', required=False, default='validation',
        help='--mode >> train, validation'
    )
    parser.add_argument('--train_path', required=False, default='./dataset/train')
    parser.add_argument('--val_path', required=False, default='./dataset/val')
    parser.add_argument('--limit', required=False, default=5)

    args = parser.parse_args()

    if args.mode == 'train':
        mode.train(args)
    elif args.mode == 'validation':
        mode.validation(args)


