import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.nn.init as init

# 假设数据已经加载和预处理
df = pd.read_csv('autism.csv')
X = df[['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5']].values
y = df['Autism'].values

# 数据预处理（标准化等）和分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量
train_features = torch.Tensor(X_train)
train_targets = torch.Tensor(y_train).long()
test_features = torch.Tensor(X_test)
test_targets = torch.Tensor(y_test).long()

# 创建数据加载器
train_dataset = TensorDataset(train_features, train_targets)
test_dataset = TensorDataset(test_features, test_targets)

train_loader = DataLoader(dataset=train_dataset, batch_size=104, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=104, shuffle=False)


class MLP(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, output_size=2, weight_variance=None, activation='relu',
                 leaky_relu_slope=0.01):
        super(MLP, self).__init__()

        # Choose the activation function based on the provided activation type
        if activation.lower() == 'relu':
            activation_function = nn.ReLU()
        elif activation.lower() == 'leaky_relu':
            activation_function = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        else:
            raise ValueError(f'Unsupported activation function: {activation}')

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation_function,
            nn.Linear(hidden_size, output_size),
        )

        if weight_variance is not None:
            # Initialize input-hidden layer weights with Gaussian distribution
            init.normal_(self.layers[0].weight, mean=0.0, std=weight_variance ** 0.5)
            init.constant_(self.layers[0].bias, 0)

        # Initialize output layer weights with the inverse of the hidden layer's width
        init.constant_(self.layers[2].weight, 1 / hidden_size)
        init.constant_(self.layers[2].bias, 0)

    def forward(self, x):
        x = self.layers(x)
        return x

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Stable Rank
def stable_rank(matrix):
    squared_singular_values = np.square(np.linalg.svd(matrix, compute_uv=False))
    return np.sum(squared_singular_values) / np.max(squared_singular_values)


def train(model, train_loader, criterion, optimizer, device):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        first_layer_weights = model.layers[0].weight.cpu().detach().numpy()
        calculated_stable_rank = stable_rank(first_layer_weights)

    print(f'Stable Rank: {calculated_stable_rank:.4f}')

    return calculated_stable_rank

# Evaluation loop
def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy


# Model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()


# %%
def main_loop(weight_variances, num_epochs, train_loader, test_loader, device):
    rank_records = {}
    test_accuracy_records = {}

    for weight_variance in weight_variances:
        print(f"Training with weight_variance = {weight_variance}")
        model = MLP(hidden_size=1000, weight_variance=weight_variance, activation='relu').to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0)
        rank_record = []
        test_accuracy_record = []

        with torch.no_grad():
            first_layer_weights = model.layers[0].weight.cpu().detach().numpy()
            calculated_stable_rank = stable_rank(first_layer_weights)
        rank_record.append(calculated_stable_rank)
        test_loss, accuracy = evaluate(model, test_loader, criterion, device)
        test_accuracy_record.append(accuracy)

        for epoch in range(1, num_epochs + 1):
            calculated_stable_rank = train(model, train_loader, criterion, optimizer, device)
            test_loss, accuracy = evaluate(model, test_loader, criterion, device)
            rank_record.append(calculated_stable_rank)
            test_accuracy_record.append(accuracy)
            print(f'Epoch: {epoch}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

        rank_records[weight_variance] = rank_record
        test_accuracy_records[weight_variance] = test_accuracy_record

    return rank_records, test_accuracy_records


import matplotlib.pyplot as plt


def plot_stable_ranks_and_test_errors_from_runs(all_rank_records, all_test_error_records, fig_filename=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # Compute means and standard deviations across runs
    weight_variances = list(all_rank_records[0].keys())
    num_epochs = len(all_rank_records[0][weight_variances[0]])
    rank_means = {wv: np.zeros(num_epochs) for wv in weight_variances}
    rank_stds = {wv: np.zeros(num_epochs) for wv in weight_variances}
    test_error_means = {wv: np.zeros(num_epochs) for wv in weight_variances}
    test_error_stds = {wv: np.zeros(num_epochs) for wv in weight_variances}

    for wv in weight_variances:
        rank_matrix = np.array([rank_records[wv] for rank_records in all_rank_records])
        test_error_matrix = np.array([test_error_records[wv] for test_error_records in all_test_error_records])
        rank_means[wv] = np.mean(rank_matrix, axis=0)
        rank_stds[wv] = np.std(rank_matrix, axis=0)
        test_error_means[wv] = np.mean(test_error_matrix, axis=0)
        test_error_stds[wv] = np.std(test_error_matrix, axis=0)

    # Plot means and shaded regions
    for weight_variance in weight_variances:
        ax1.plot(rank_means[weight_variance], label=f'Weight Variance = {weight_variance}')
        ax1.fill_between(range(num_epochs), rank_means[weight_variance] - 3 * rank_stds[weight_variance],
                         rank_means[weight_variance] + rank_stds[weight_variance], alpha=0.3)
        ax2.plot(test_error_means[weight_variance], label=f'Weight Variance = {weight_variance}')
        ax2.fill_between(range(num_epochs), test_error_means[weight_variance] - 3 * test_error_stds[weight_variance],
                         test_error_means[weight_variance] + test_error_stds[weight_variance], alpha=0.3)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Stable Rank')
    ax1.set_title('Stable Ranks for Different Weight Variances')
    ax1.legend()
    ax1.grid()

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Test Accuracy for Different Weight Variances')
    ax2.legend()
    ax2.grid()

    if fig_filename:
        plt.savefig(fig_filename)

    plt.show()



import matplotlib.pyplot as plt


def plot_stable_ranks_and_test_errors_from_runs(all_rank_records, all_test_error_records, fig_filename=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # Compute means and standard deviations across runs
    weight_variances = list(all_rank_records[0].keys())
    num_epochs = len(all_rank_records[0][weight_variances[0]])
    rank_means = {wv: np.zeros(num_epochs) for wv in weight_variances}
    rank_stds = {wv: np.zeros(num_epochs) for wv in weight_variances}
    test_error_means = {wv: np.zeros(num_epochs) for wv in weight_variances}
    test_error_stds = {wv: np.zeros(num_epochs) for wv in weight_variances}

    for wv in weight_variances:
        rank_matrix = np.array([rank_records[wv] for rank_records in all_rank_records])
        test_error_matrix = np.array([test_error_records[wv] for test_error_records in all_test_error_records])
        rank_means[wv] = np.mean(rank_matrix, axis=0)
        rank_stds[wv] = np.std(rank_matrix, axis=0)
        test_error_means[wv] = np.mean(test_error_matrix, axis=0)
        test_error_stds[wv] = np.std(test_error_matrix, axis=0)

    # Plot means and shaded regions
    for weight_variance in weight_variances:
        ax1.plot(rank_means[weight_variance], label=f'Weight Variance = {weight_variance}')
        ax1.fill_between(range(num_epochs), rank_means[weight_variance] - 3 * rank_stds[weight_variance],
                         rank_means[weight_variance] + rank_stds[weight_variance], alpha=0.003)
        ax2.plot(test_error_means[weight_variance], label=f'Weight Variance = {weight_variance}')
        ax2.fill_between(range(num_epochs), test_error_means[weight_variance] - 3 * test_error_stds[weight_variance],
                         test_error_means[weight_variance] + test_error_stds[weight_variance], alpha=0.03)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Stable Rank')
    ax1.set_title('Stable Ranks for Different Weight Variances')
    ax1.legend()
    ax1.grid()

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Test Accuracy for Different Weight Variances')
    ax2.legend()
    ax2.grid()

    if fig_filename:
        plt.savefig(fig_filename)

    plt.show()

num_runs = 1
num_epochs = 5000  # Change this to the desired number of epochs
criterion = nn.CrossEntropyLoss()  # Add criterion definition
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_variances = [0.00001, 0.00005, 0.0001, 0.0005, 0.001]  # Add more variances if desired
all_rank_records = []
all_test_error_records = []

for run in range(num_runs):
    print(f"Run {run + 1}/{num_runs}")
    seed = 42 + run  # Set a different seed for each run
    set_seed(seed)
    rank_records, test_error_records = main_loop(weight_variances, num_epochs, train_loader, test_loader, device)
    all_rank_records.append(rank_records)
    all_test_error_records.append(test_error_records)
    print()

# Plot the stable ranks and test errors from multiple runs and save the figure as a PNG
plot_stable_ranks_and_test_errors_from_runs(all_rank_records, all_test_error_records, 'autism_stable_ranks_and_test_errors_relu.png')

import pickle

def save_records(rank_records, test_error_records, rank_filename, test_error_filename):
    with open(rank_filename, 'wb') as rank_file:
        pickle.dump(rank_records, rank_file)

    with open(test_error_filename, 'wb') as test_error_file:
        pickle.dump(test_error_records, test_error_file)


# Save the records
save_records(all_rank_records, all_test_error_records, 'autism_rank_records_relu.pickle', 'autism_test_error_records_relu.pickle')
