import argparse
import numpy as np
import pandas as pd
import torch

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST, KMNIST, QMNIST
from torchvision.models import vgg11_bn

plt.rcParams['font.size'] = 12
plt.rcParams['savefig.format'] = 'pdf'

dataset_list = [MNIST, FashionMNIST, KMNIST, QMNIST]
dataset_name_list = [dataset.__name__ for dataset in dataset_list]
num_fc_list = [9216, 9216, 9216, 9216, 9216, 12544, 12544, 135424, 12544]

mean_std_list = [
        ((0.1307,), (0.3081,)),
        ((0.1307,), (0.3081,)),
        ((0.1307,), (0.3081,)),
        ((0.1307,), (0.3081,)),
        ]

train_range_list = [
        range(50000), 
        range(50000),
        range(50000),
        range(50000),
        ]

validation_range_list = [
        range(50000, 60000),
        range(50000, 60000),
        range(50000, 60000),
        range(50000, 60000),
        ]

test_range_list = [
        range(10000),
        range(10000),
        range(10000),
        range(10000),
        ]

def save_loss(train_loss, validation_loss, dataset_name, results_dir):
    fig, ax = plt.subplots()
    plt.grid(True)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.ylim([0, 1])
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.plot(train_loss)
    plt.plot(validation_loss)
    ax.tick_params(axis='both', which='major', labelsize='large')
    ax.tick_params(axis='both', which='minor', labelsize='large')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(f'{results_dir}/{dataset_name}-loss', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # DO NOT EDIT BLOCK
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--full', default=False, action='store_true')
    args = parser.parse_args()

    cache_dir = 'cache'
    results_dir = 'results'
    # END OF DO NOT EDIT BLOCK

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.full:
        num_epochs = 50
    else:
        num_epochs = 1
        train_range_list = [train_range[:10] for train_range in train_range_list]
        validation_range_list = [validation_range[:10] for validation_range in validation_range_list]
        test_range_list = [test_range[:10] for test_range in test_range_list]
    lr = 0.01
    batch_size = 64
    train_loss_array = np.zeros((len(dataset_list), num_epochs))
    validation_loss_array = np.zeros((len(dataset_list), num_epochs))
    test_accuracy_array = np.zeros((len(dataset_list)))
    test_batch_size = 1000
    criterion = nn.CrossEntropyLoss()
    for index_dataset, (dataset, dataset_name, train_range, validation_range, test_range, mean_std, num_fc) in enumerate(zip(dataset_list, dataset_name_list, train_range_list, validation_range_list, test_range_list, mean_std_list, num_fc_list)):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
            transforms.Normalize(mean_std[0], mean_std[1])
            ])
        train_dataset = dataset(cache_dir, train=True, transform=transform, download=True)
        validation_dataset = dataset(cache_dir, train=True, transform=transform)
        test_dataset = dataset(cache_dir, train=False, transform=transform, download=True)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_range))
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(validation_range))
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, sampler=SubsetRandomSampler(test_range))
        num_classes = len(train_dataset.classes)
        model = vgg11_bn().to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        validation_loss_best = float('inf')
        model_path = f'{results_dir}/{dataset.__name__}.pt'
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
            train_loss_array[index_dataset, index_epoch] = train_loss
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
            validation_loss_array[index_dataset, index_epoch] = validation_loss
            accuracy = 100. * correct / total
            print(f'{dataset_name}, Epoch: {epoch}, Validation average loss: {validation_loss:.4f}, Validation accuracy: {accuracy:.2f}%')
            if validation_loss < validation_loss_best:
                validation_loss_best = validation_loss
                torch.save(model.state_dict(), model_path)
                print('saving as best model')
        model = vgg11_bn().to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_dataloader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                prediction = output.argmax(dim=1, keepdim=True)
                correct += prediction.eq(target.view_as(prediction)).sum().item()
                total += output.shape[0]
            accuracy = 100. * correct / total
            test_accuracy_array[index_dataset] = accuracy
            print(f'{dataset_name}, Test accuracy: {accuracy:.2f}%')

    df_keys_values = pd.DataFrame({'key': [
        'num_epochs',
        'batch_size',
        'lr'],
        'value': [
            str(int(num_epochs)),
            str(int(batch_size)),
            lr]})
    df_keys_values.to_csv(f'{results_dir}/keys-values.csv')

    for index_dataset_name, (dataset_name, train_loss, validation_loss) in enumerate(zip(dataset_name_list, train_loss_array, validation_loss_array)):
        save_loss(train_loss, validation_loss, dataset_name, results_dir)

    max_per_column_list = test_accuracy_array.max(0)
    df = pd.DataFrame(test_accuracy_array)
    df.to_latex(f'{results_dir}/metrics.tex', bold_rows=True, column_format='r|r', multirow=True, escape=False)
