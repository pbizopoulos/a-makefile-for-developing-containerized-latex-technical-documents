import argparse
import numpy as np
import os
import pandas as pd
import torch

from matplotlib import pyplot as plt
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import MNIST

plt.rcParams['font.size'] = 12
plt.rcParams['savefig.format'] = 'pdf'


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(2304, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    # Set random seeds.
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', default=False, action='store_true')
    parser.add_argument('--use-cuda', default=False, action='store_true')
    parser.add_argument('--cache-dir')
    parser.add_argument('--results-dir')
    args = parser.parse_args()
    device = torch.device('cuda' if (torch.cuda.is_available() and args.use_cuda) else 'cpu')
    if args.full:
        num_epochs = 10
        train_range = range(40000)
        validation_range = range(40000, 50000)
        test_range = range(10000)
    else:
        num_epochs = 1
        train_range = range(10)
        validation_range = range(10)
        test_range = range(10)

    lr = 0.01
    batch_size = 64
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = MNIST(args.cache_dir, train=True, transform=transform, download=True)
    validation_dataset = MNIST(args.cache_dir, train=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_range))
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(validation_range))
    model = CustomModel()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loss_array = np.zeros((num_epochs))
    validation_loss_array = np.zeros((num_epochs))
    validation_loss_best = float('inf')
    model_path = f'{args.results_dir}/model_best.pt'
    for index_epoch, epoch in enumerate(range(num_epochs)):
        train_loss_sum = 0
        model.train()
        for data, target in train_dataloader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
        train_loss = train_loss_sum / len(train_dataloader)
        train_loss_array[index_epoch] = train_loss
        model.eval()
        validation_loss_sum = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in validation_dataloader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                validation_loss_sum += criterion(output, target).item()
                prediction = output.argmax(dim=1, keepdim=True)
                correct += prediction.eq(target.view_as(prediction)).sum().item()
                total += output.shape[0]
        validation_loss = validation_loss_sum / len(validation_dataloader)
        validation_loss_array[index_epoch] = validation_loss
        validation_accuracy = 100. * correct / total
        print(f'Epoch: {epoch}, Validation average loss: {validation_loss:.4f}, Validation accuracy: {validation_accuracy:.2f}%')
        if validation_loss < validation_loss_best:
            validation_loss_best = validation_loss
            validation_accuracy_best = validation_accuracy
            torch.save(model.state_dict(), model_path)
            print('saving as best model')

    # Create pdf images.
    plt.figure(constrained_layout=True, figsize=(6, 2))
    plt.plot(train_loss_array)
    plt.plot(validation_loss_array)
    plt.grid(True)
    plt.xlabel('Loss')
    plt.ylabel('Epochs')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.savefig(f'{args.results_dir}/image')
    plt.close()

    # Create tables.
    table = [train_loss_array[-1], validation_loss_array[-1], validation_accuracy_best]
    df = pd.DataFrame(table)
    df.to_latex(f'{args.results_dir}/table.tex', float_format="%.2f")

    # Create variables.
    df = pd.DataFrame({'key': ['lr', 'batch_size', 'validation_accuracy_best'], 'value': [lr, batch_size, validation_accuracy_best]})
    df.to_csv(f'{args.results_dir}/keys-values.csv', index=False, float_format='%.1f')
