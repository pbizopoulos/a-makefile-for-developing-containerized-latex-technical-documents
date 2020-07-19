import argparse
import numpy as np
import os
import pandas as pd
import torch

from matplotlib import pyplot as plt

plt.rcParams['font.size'] = 12
plt.rcParams['savefig.format'] = 'pdf'


if __name__ == '__main__':
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', default=False, action='store_true')
    parser.add_argument('--cache-dir', default='cache')
    parser.add_argument('--results-train-dir', default='results-train')
    parser.add_argument('--results-dir', default='results')
    args = parser.parse_args()
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.full:
        num_samples = 200
    else:
        num_samples = 20

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
    plt.savefig(f'{args.results_dir}/image')
    plt.close()

    # Creating tables
    num_columns = 11
    table = np.random.random((3, num_columns))
    df = pd.DataFrame(table)
    df.to_latex(f'{args.results_dir}/table.tex', float_format="%.2f")

    # Creating variables
    df = pd.DataFrame({'key': ['num_samples', 'num_columns', 'mu', 'sigma'], 'value': [num_samples, num_columns, mu, sigma]})
    df.to_csv(f'{args.results_dir}/keys-values.csv', index=False, float_format='%.1f')
