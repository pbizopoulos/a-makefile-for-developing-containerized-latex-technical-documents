#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from pathlib import Path

plt.rcParams.update({'font.size': 12})

if __name__ == '__main__':
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    args = parser.parse_args()
    if args.full:
        num_samples = 200
    else:
        num_samples = 200
    if args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'
    path_paper = './paper'
    path_images = f'{path_paper}/images'
    path_tables = f'{path_paper}/tables'
    Path(path_images).mkdir(parents=True, exist_ok=True)
    Path(path_tables).mkdir(parents=True, exist_ok=True)

    # Creating pdf images
    mu = 0.42
    sigma = 0.69
    data = mu + sigma*np.random.randn(num_samples) + np.linspace(0, 5, num_samples)
    plt.figure(constrained_layout=True, figsize=(6, 2))
    plt.plot(data)
    plt.grid(True)
    plt.xlabel('X label')
    plt.ylabel('Y label')
    plt.ylim([-2, 7])
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.savefig(f'{path_images}/image.pdf')
    plt.close()

    # Creating tables
    num_columns = 11
    table = np.random.random((3, num_columns))
    df = pd.DataFrame(table)
    df.to_latex(f'{path_tables}/table.tex', float_format="%.2f")

    # Creating variables
    df = pd.DataFrame({'key': ['num_samples', 'num_columns', 'mu', 'sigma'], 'value': [num_samples, num_columns, mu, sigma]})
    df.to_csv(f'{path_paper}/keys_values.csv', index=False, float_format='%.1f')
