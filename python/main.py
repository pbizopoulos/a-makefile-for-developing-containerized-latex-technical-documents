from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from os import environ
from os.path import join
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import KMNIST, MNIST, QMNIST, FashionMNIST
from torchvision.models import mobilenet_v2
from torchvision.transforms import Compose, Lambda, Normalize, ToTensor
from torchvision.utils import save_image
import numpy as np
import pandas as pd
import torch


def change_module(model, module_new, module_old):
    for (child_name, child) in model.named_children():
        if isinstance(child, module_old):
            setattr(model, child_name, module_new())
        else:
            change_module(child, module_new, module_old)


def main():
    artifacts_dir = environ['ARTIFACTSDIR']
    full = environ['FULL']
    plt.rcParams['font.size'] = 18
    plt.rcParams['savefig.format'] = 'pdf'
    dataset_list = [MNIST, FashionMNIST, KMNIST, QMNIST]
    dataset_name_list = [dataset.__name__ for dataset in dataset_list]
    activation_function_list = ['ReLU', 'ReLU6', 'SiLU']
    mean_std_list = [((0.1307,), (0.3081,)), ((0.1307,), (0.3081,)), ((0.1307,), (0.3081,)), ((0.1307,), (0.3081,))]
    epochs_num = 20
    training_range_list = [range(50000), range(50000), range(50000), range(50000)]
    validation_range_list = [range(50000, 60000), range(50000, 60000), range(50000, 60000), range(50000, 60000)]
    test_range_list = [range(10000), range(10000), range(10000), range(10000)]
    if not full:
        epochs_num = 2
        training_range_list = [training_range[:10] for training_range in training_range_list]
        validation_range_list = [validation_range[:10] for validation_range in validation_range_list]
        test_range_list = [test_range[:10] for test_range in test_range_list]
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = 0.01
    batch_size = 64
    training_loss_array = np.zeros((len(dataset_list), len(activation_function_list), epochs_num))
    validation_loss_array = np.zeros((len(dataset_list), len(activation_function_list), epochs_num))
    test_accuracy_array = np.zeros((len(dataset_list), len(activation_function_list)))
    test_batch_size = 1000
    criterion = nn.CrossEntropyLoss()
    for (dataset_index, (dataset, dataset_name, training_range, validation_range, test_range, mean_std)) in enumerate(zip(dataset_list, dataset_name_list, training_range_list, validation_range_list, test_range_list, mean_std_list)):
        transform = Compose([ToTensor(), Lambda(lambda x: torch.cat([x, x, x], 0)), Normalize(mean_std[0], mean_std[1])])
        training_dataset = dataset(artifacts_dir, transform=transform, download=True)
        test_dataset = dataset(artifacts_dir, train=False, transform=transform, download=True)
        training_dataloader = DataLoader(training_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(training_range))
        validation_dataloader = DataLoader(training_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(validation_range))
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, sampler=SubsetRandomSampler(test_range))
        for (activation_function_index, activation_function) in enumerate(activation_function_list):
            model = mobilenet_v2().to(device)
            if activation_function == 'SiLU':
                change_module(model, nn.SiLU, nn.ReLU6)
            elif activation_function == 'ReLU':
                change_module(model, nn.ReLU, nn.ReLU6)
            optimizer = optim.SGD(model.parameters(), lr=lr)
            validation_loss_best = float('inf')
            model_file_name = f'{dataset.__name__}-{activation_function}'
            model_file_path = join(artifacts_dir, f'{model_file_name}.pt')
            for epoch in range(epochs_num):
                training_loss_sum = 0
                model.train()
                for (data, target) in training_dataloader:
                    data = data.to(device)
                    target = target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    training_loss_sum += loss.item()
                training_loss = training_loss_sum / len(training_dataloader)
                training_loss_array[dataset_index, activation_function_index, epoch] = training_loss
                model.eval()
                validation_loss_sum = 0
                correct = 0
                total = 0
                with torch.no_grad():
                    for (data, target) in validation_dataloader:
                        data = data.to(device)
                        target = target.to(device)
                        output = model(data)
                        validation_loss_sum += criterion(output, target).item()
                        prediction = output.argmax(dim=1, keepdim=True)
                        correct += prediction.eq(target.view_as(prediction)).sum().item()
                        total += output.shape[0]
                validation_loss = validation_loss_sum / len(validation_dataloader)
                validation_loss_array[dataset_index, activation_function_index, epoch] = validation_loss
                accuracy = 100.0 * correct / total
                if validation_loss < validation_loss_best:
                    validation_loss_best = validation_loss
                    torch.save(model.state_dict(), model_file_path)
            model = mobilenet_v2().to(device)
            if activation_function == 'SiLU':
                change_module(model, nn.SiLU, nn.ReLU6)
            elif activation_function == 'ReLU':
                change_module(model, nn.ReLU, nn.ReLU6)
            model.load_state_dict(torch.load(model_file_path))
            model.eval()
            kernels = model.features[0][0].weight.detach().clone()
            save_image(kernels[:25], join(artifacts_dir, f'{model_file_name}-kernels.pdf'), padding=1, nrow=5, normalize=True)
            correct = 0
            total = 0
            with torch.no_grad():
                for (data, target) in test_dataloader:
                    data = data.to(device)
                    target = target.to(device)
                    output = model(data)
                    prediction = output.argmax(dim=1, keepdim=True)
                    correct += prediction.eq(target.view_as(prediction)).sum().item()
                    total += output.shape[0]
                accuracy = 100.0 * correct / total
                test_accuracy_array[dataset_index, activation_function_index] = accuracy
    keys_values_df = pd.DataFrame({'key': ['epochs-num', 'batch-size', 'lr'], 'value': [str(int(epochs_num)), str(int(batch_size)), lr]})
    keys_values_df.to_csv(join(artifacts_dir, 'keys-values.csv'))
    for (dataset_name, training_loss, validation_loss) in zip(dataset_name_list, training_loss_array, validation_loss_array):
        (_, ax) = plt.subplots()
        plt.grid(True)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.ylim([0, 1])
        plt.xlabel('Epochs', fontsize=18)
        if dataset_name == 'MNIST':
            plt.ylabel('loss', fontsize=18)
        for (training_loss_, validation_loss_, activation_function, color) in zip(training_loss, validation_loss, activation_function_list, ['b', 'orange']):
            plt.plot(training_loss_, label=f'Training {activation_function}', color=color)
            plt.plot(validation_loss_, label=f'Validation {activation_function}', linestyle='--', color=color)
        plt.title(dataset_name)
        ax.tick_params(axis='both', which='major', labelsize='large')
        ax.tick_params(axis='both', which='minor', labelsize='large')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        plt.savefig(join(artifacts_dir, f'{dataset_name}-loss'), bbox_inches='tight')
        plt.close()
    styler = pd.DataFrame(test_accuracy_array.T, index=activation_function_list, columns=dataset_name_list).style
    styler.format(precision=2)
    styler.highlight_max(props='bfseries: ;')
    styler.to_latex(join(artifacts_dir, 'metrics.tex'), hrules=True)


if __name__ == '__main__':
    main()
