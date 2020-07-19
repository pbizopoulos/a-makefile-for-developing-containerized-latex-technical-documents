import argparse
import numpy as np
import os
import pandas as pd
import torch


def set_seed():
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)


if __name__ == '__main__':
    set_seed()
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', default=False, action='store_true')
    parser.add_argument('--cache-dir', default='cache')
    parser.add_argument('--results-train-dir', default='results-train')
    args = parser.parse_args()
    if not os.path.exists(args.cache_dir):
        os.mkdir(args.cache_dir)
    if not os.path.exists(args.results_train_dir):
        os.mkdir(args.results_train_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.full:
        num_samples = 200
    else:
        num_samples = 20
