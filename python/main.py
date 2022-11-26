from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from os import environ
from os.path import join
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import FashionMNIST, KMNIST, MNIST, QMNIST
from torchvision.models import mobilenet_v2
from torchvision.transforms import Compose, Lambda, Normalize, ToTensor
from torchvision.utils import save_image
import hashlib
import numpy as np
import pandas as pd
import torch


def change_module(model, module_new, module_old):
    for child_name, child in model.named_children():
        if isinstance(child, module_old):
            setattr(model, child_name, module_new())
        else:
            change_module(child, module_new, module_old)


def main():
    plt.rcParams['font.size'] = 18
    plt.rcParams['savefig.format'] = 'pdf'
    dataset_list = [MNIST, FashionMNIST, KMNIST, QMNIST]
    dataset_name_list = [dataset.__name__ for dataset in dataset_list]
    activation_function_name_list = ['ReLU', 'ReLU6', 'SiLU']
    std_mean_list = [((0.1307,), (0.3081,)), ((0.1307,), (0.3081,)), ((0.1307,), (0.3081,)), ((0.1307,), (0.3081,))]
    epochs_num = 20
    range_training_list = [range(50000), range(50000), range(50000), range(50000)]
    range_validation_list = [range(50000, 60000), range(50000, 60000), range(50000, 60000), range(50000, 60000)]
    test_range_list = [range(10000), range(10000), range(10000), range(10000)]
    if environ['DEBUG'] == '1':
        epochs_num = 2
        range_training_list = [range_training[:10] for range_training in range_training_list]
        range_validation_list = [range_validation[:10] for range_validation in range_validation_list]
        test_range_list = [test_range[:10] for test_range in test_range_list]
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = 0.01
    batch_size = 64
    loss_training_array = np.zeros((len(dataset_list), len(activation_function_name_list), epochs_num))
    loss_validation_array = np.zeros((len(dataset_list), len(activation_function_name_list), epochs_num))
    accuracy_test_array = np.zeros((len(dataset_list), len(activation_function_name_list)))
    batch_size_test = 1000
    cross_entropy_loss = nn.CrossEntropyLoss()
    for dataset_index, (dataset, dataset_name, range_training, range_validation, test_range, std_mean) in enumerate(zip(dataset_list, dataset_name_list, range_training_list, range_validation_list, test_range_list, std_mean_list)):
        transform = Compose([ToTensor(), Lambda(lambda tensor: torch.cat([tensor, tensor, tensor], 0)), Normalize(std_mean[0], std_mean[1])])
        dataset_training = dataset('bin', transform=transform, download=True)
        dataset_test = dataset('bin', train=False, transform=transform, download=True)
        dataloader_training = DataLoader(dataset_training, batch_size=batch_size, sampler=SubsetRandomSampler(range_training))
        dataloader_validation = DataLoader(dataset_training, batch_size=batch_size, sampler=SubsetRandomSampler(range_validation))
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size_test, sampler=SubsetRandomSampler(test_range))
        for activation_function_name_index, activation_function_name in enumerate(activation_function_name_list):
            model = mobilenet_v2().to(device)
            if activation_function_name == 'SiLU':
                change_module(model, nn.SiLU, nn.ReLU6)
            elif activation_function_name == 'ReLU':
                change_module(model, nn.ReLU, nn.ReLU6)
            optimizer = optim.SGD(model.parameters(), lr=lr)
            loss_validation_best = float('inf')
            model_file_name = f'{dataset.__name__}-{activation_function_name}'
            model_file_path = join('bin', f'{model_file_name}.pt')
            for epoch in range(epochs_num):
                loss_training_sum = 0
                predictions_num = 0
                model.train()
                for data, target in dataloader_training:
                    data = data.to(device)
                    target = target.to(device)
                    output = model(data)
                    predictions_num += output.shape[0]
                    loss = cross_entropy_loss(output, target)
                    loss_training_sum += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                loss_training = loss_training_sum / predictions_num
                loss_training_array[dataset_index, activation_function_name_index, epoch] = loss_training
                loss_validation_sum = 0
                predictions_num = 0
                model.eval()
                with torch.no_grad():
                    for data, target in dataloader_validation:
                        data = data.to(device)
                        target = target.to(device)
                        output = model(data)
                        predictions_num += output.shape[0]
                        loss_validation_sum += cross_entropy_loss(output, target).item()
                loss_validation = loss_validation_sum / predictions_num
                loss_validation_array[dataset_index, activation_function_name_index, epoch] = loss_validation
                if loss_validation < loss_validation_best:
                    loss_validation_best = loss_validation
                    torch.save(model.state_dict(), model_file_path)
            model = mobilenet_v2().to(device)
            if activation_function_name == 'SiLU':
                change_module(model, nn.SiLU, nn.ReLU6)
            elif activation_function_name == 'ReLU':
                change_module(model, nn.ReLU, nn.ReLU6)
            model.load_state_dict(torch.load(model_file_path))
            kernels = model.features[0][0].weight.detach().clone()
            save_image(kernels[:25], join('bin', f'{model_file_name}-kernels.pdf'), padding=1, nrow=5, normalize=True)
            predictions_correct_num = 0
            predictions_num = 0
            model.eval()
            with torch.no_grad():
                for data, target in dataloader_test:
                    data = data.to(device)
                    target = target.to(device)
                    output = model(data)
                    prediction = output.argmax(dim=1)
                    predictions_correct_num += sum(prediction == target).item()
                    predictions_num += output.shape[0]
                accuracy_test_array[dataset_index, activation_function_name_index] = 100.0 * predictions_correct_num / predictions_num
    keys_values_df = pd.DataFrame({'key': ['epochs-num', 'batch-size', 'lr'], 'value': [str(int(epochs_num)), str(int(batch_size)), lr]})
    keys_values_df.to_csv(join('bin', 'keys-values.csv'))
    for dataset_name, loss_training, loss_validation in zip(dataset_name_list, loss_training_array, loss_validation_array):
        _, ax = plt.subplots()
        plt.grid(True)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.ylim([0, 1])
        plt.xlabel('Epochs', fontsize=18)
        if dataset_name == 'MNIST':
            plt.ylabel('loss', fontsize=18)
        for loss_training_, loss_validation_, activation_function_name, color in zip(loss_training, loss_validation, activation_function_name_list, ['b', 'orange']):
            plt.plot(loss_training_, label=f'Training {activation_function_name}', color=color)
            plt.plot(loss_validation_, label=f'Validation {activation_function_name}', linestyle='--', color=color)
        plt.title(dataset_name)
        ax.tick_params(axis='both', which='major', labelsize='large')
        ax.tick_params(axis='both', which='minor', labelsize='large')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        plt.savefig(join('bin', f'{dataset_name}-loss'), bbox_inches='tight')
        plt.close()
    styler = pd.DataFrame(accuracy_test_array.T, index=activation_function_name_list, columns=dataset_name_list).style
    styler.format(precision=2)
    styler.highlight_max(props='bfseries: ;')
    styler.to_latex(join('bin', 'metrics.tex'), hrules=True)
    if environ['DEBUG'] == '1':
        with open(join('bin', 'metrics.tex'), 'rb') as file:
            assert hashlib.sha256(file.read()).hexdigest() == 'aaec2ea032278b725dfdc212b2ce40c120b1a76c077b1bf7f9c7e5a50216f6e1'


if __name__ == '__main__':
    main()
