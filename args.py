import argparse


def parse_codesearch_args(mode='train'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--data_dir', type=str,
                        default='datasets/CodeSearchNet')
    parser.add_argument('-lang', '--language', type=str, default='javascript')
    parser.add_argument('-ptm', '--pretrained_model',
                        type=str, default='microsoft/codebert-base')
    parser.add_argument('-bs', '--batch_size', type=int, default=32)

    if mode == 'train':
        return training_args(parser)
    elif mode == 'eval':
        return evaluation_args(parser)

    return parser.parse_args()


# TODO: edit this method
def parse_code2test_args(mode='train'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--data_dir', type=str,
                        default='datasets/methods2test/corpus/raw/fm/')
    parser.add_argument('-ptm', '--pretrained_model', type=str,
                        default='Salesforce/codet5-small')
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('--prefix', action='store_true')

    if mode == 'train':
        return training_args(parser)
    elif mode == 'eval':
        return evaluation_args(parser)

    return parser.parse_args()


def training_args(parser):
    parser.add_argument('-out', '--output_dir',
                        type=str, default='checkpoints')
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-s', '--scheduler', type=str, default='step',
                        choices=['step', 'plateau', 'linear', 'cosine'])
    return parser.parse_args()


def evaluation_args(parser):
    parser.add_argument('-ckpt', '--checkpoint_path', type=str, default=None)
    return parser.parse_args()
