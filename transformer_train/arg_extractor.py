import argparse
import json
import os
import sys
import GPUtil

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(
        description='Welcome to the MLP course\'s Pytorch training and inference helper script')

    parser.add_argument('--batch_size', nargs="?", type=int, default=4, help='Batch_size for experiment')
    # parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=-1, help='Batch_size for experiment')
    parser.add_argument('--target', type=str, help='Dataset on which the system will train/eval our model')
    # parser.add_argument('--seed', nargs="?", type=int, default=7112018,
    #                     help='Seed to use for random number generator for experiment')

    # parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
    #                     metavar='LR', help='initial learning rate')
    # parser.add_argument('--dropout',  default=0.4, type=float,
    #                     help='dropout rate')
    parser.add_argument('--num_epochs', nargs="?", type=int, default=5, help='The experiment\'s epoch budget')
    parser.add_argument('--freeze_embeddings', nargs="?", type=int, default=0, help='The experiment\'s epoch budget')
    # parser.add_argument('--num_hidden', nargs="?", type=int, default=128)
    parser.add_argument('--addname', nargs="?", type=str, default="forzen",
                        help='Experiment name - to be used for building the experiment folder')
    parser.add_argument('--experiment_name', nargs="?", type=str, default="exp_1",
                        help='Experiment name - to be used for building the experiment folder')
    # parser.add_argument('--stopwords', nargs="?", type=str2bool, default=True,
    #                     help='stop words')

    # parser.add_argument('--result_name', nargs="?", type=str,
    #                     help='Experiment name - to be used for building the experiment folder')
    # parser.add_argument('--update_emb', nargs="?", type=str2bool, default=True,
    #                     help='updating embedding flag')
    args = parser.parse_args()



    return args



