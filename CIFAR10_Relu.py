import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pickle

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 设置随机种子以确保可重现性
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MLP(nn.Module):
    def __init__(self, input_size=3 * 32 * 32, hidden_size=512, output_size=10, weight_variance=None, activation='relu',
                 leaky_relu_slope=0.01):
        super(MLP, self).__init__()
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

        # Initialize the first layer if a weight variance is specified
        if weight_variance is not None:
            init.normal_(self.layers[0].weight, mean=0.0, std=weight_variance ** 0.5)
            init.constant_(self.layers[0].bias, 0)

        # Initialize output layer weights with the inverse of the hidden layer's width
        init.constant_(self.layers[2].weight, 1 / hidden_size)
        init.constant_(self.layers[2].bias, 0)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        return self.layers(x)

# CIFAR10数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=50000, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=50000, shuffle=False)

# 稳定等级函数
def stable_rank(matrix):
    squared_singular_values = np.square(np.linalg.svd(matrix, compute_uv=False))
    return np.sum(squared_singular_values) / np.max(squared_singular_values)

# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    # 计算第一层权重的稳定等级
    with torch.no_grad():
        first_layer_weights = model.layers[0].weight.cpu().numpy()
        sr = stable_rank(first_layer_weights)
    return sr

# 评估函数
def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # 获取最大概率的索引
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

# 主循环
def main_loop(weight_variances, num_epochs, train_loader, test_loader, device):
    rank_records = {}
    test_accuracy_records = {}
    for weight_variance in weight_variances:
        print(f"Training with weight variance = {weight_variance}")
        model = MLP(weight_variance=weight_variance).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0)
        criterion = nn.CrossEntropyLoss()
        rank_record = []
        test_accuracy_record = []
        for epoch in range(num_epochs):
            sr = train(model, train_loader, criterion, optimizer, device)
            test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
            print(f'Epoch: {epoch}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
            rank_record.append(sr)
            test_accuracy_record.append(test_accuracy)
        rank_records[weight_variance] = rank_record
        test_accuracy_records[weight_variance] = test_accuracy_record
    return rank_records, test_accuracy_records

# 绘制函数
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


num_runs = 1
num_epochs = 100  # Change this to the desired number of epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_variances = [0.00001, 0.00005, 0.0001, 0.0005, 0.001]
all_rank_records = []
all_test_error_records = []

for run in range(num_runs):
    print(f"Run {run + 1}/{num_runs}")
    seed = 42 + run
    set_seed(seed)
    rank_records, test_error_records = main_loop(weight_variances, num_epochs, train_loader, test_loader, device)
    all_rank_records.append(rank_records)
    all_test_error_records.append(test_error_records)
    print()

# Plot the stable ranks and test errors from multiple runs and save the figure as a PNG
plot_stable_ranks_and_test_errors_from_runs(all_rank_records, all_test_error_records, 'CIFAR10_stable_ranks_and_test_errors.png')

import pickle

def save_records(rank_records, test_error_records, rank_filename, test_error_filename):
    with open(rank_filename, 'wb') as rank_file:
        pickle.dump(rank_records, rank_file)

    with open(test_error_filename, 'wb') as test_error_file:
        pickle.dump(test_error_records, test_error_file)


# Save the records
save_records(all_rank_records, all_test_error_records, 'CIFAR10_rank_records.pickle', 'CIFAR10_test_error_records.pickle')
