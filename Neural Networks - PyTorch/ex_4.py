import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

MINI_BATCH_SIZE = 64


class NeuralNetworkA(nn.Module):
    def __init__(self, image_size):
        super(NeuralNetworkA, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


class NeuralNetworkB(nn.Module):
    def __init__(self, image_size):
        super(NeuralNetworkB, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        return F.log_softmax(self.fc2(x), dim=1)


class NeuralNetworkC(nn.Module):
    def __init__(self, image_size):
        super(NeuralNetworkC, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.first_bnl = nn.BatchNorm1d(100)
        self.second_bnl = nn.BatchNorm1d(50)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.first_bnl(self.fc0(x)))
        x = F.relu(self.second_bnl(self.fc1(x)))
        return F.log_softmax(self.fc2(x), dim=1)


class NeuralNetworkD(nn.Module):
    def __init__(self, image_size):
        super(NeuralNetworkD, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return F.log_softmax(self.fc5(x), dim=1)


class NeuralNetworkE(nn.Module):
    def __init__(self, image_size):
        super(NeuralNetworkE, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = torch.sigmoid(self.fc0(x))
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return F.log_softmax(self.fc5(x), dim=1)


def calculate_loss_and_accuracy(nn_model, data, batch_size, data_name):
    loss = 0
    correct_predictions = 0

    nn_model.eval()
    for x, y in data:
        outputs = nn_model(x)
        loss += F.nll_loss(outputs, y, reduction='sum').item()
        max_arg_outputs = torch.max(outputs, 1)[1]
        correct_predictions += max_arg_outputs.eq(y.data.view_as(max_arg_outputs)).cpu().sum()

    data_length = len(data)
    loss = loss / (data_length * batch_size)
    accuracy = 100. * correct_predictions/(data_length * batch_size)
    print("{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(data_name, loss, correct_predictions, data_length * batch_size, accuracy))

    return loss, accuracy


def train_neural_network(nn_model, train_set, validation_set):
    epochs = 10
    learning_rate = 0.001

    train_set_loss_score = {}
    train_set_accuracy_score = {}
    validation_set_loss_score = {}
    validation_set_accuracy_score = {}

    optimizer = optim.Adam(nn_model.parameters(), lr=learning_rate)
    for epoch in range(epochs):

        nn_model.train()
        for x_mini_batch, y_mini_batch in train_set:
            optimizer.zero_grad()
            outputs = nn_model(x_mini_batch)
            loss = F.nll_loss(outputs, y_mini_batch)
            loss.backward()
            optimizer.step()

        print("\nEpoch number {}.".format(epoch))
        train_set_epoch_loss, train_set_epoch_accuracy = calculate_loss_and_accuracy(nn_model, train_set, MINI_BATCH_SIZE, "Train Set")
        train_set_loss_score[epoch] = train_set_epoch_loss
        train_set_accuracy_score[epoch] = train_set_epoch_accuracy

        validation_set_epoch_loss, validation_set_epoch_accuracy = calculate_loss_and_accuracy(nn_model, validation_set, 1, "Validation Set")
        validation_set_loss_score[epoch] = validation_set_epoch_loss
        validation_set_accuracy_score[epoch] = validation_set_epoch_accuracy

    # Plot graph for the loss scores.
    train_set_plot = plt.plot(list(train_set_loss_score.keys()), list(train_set_loss_score.values()), "black", label='Train Loss')
    validation_set_plot = plt.plot(list(validation_set_loss_score.keys()), list(validation_set_loss_score.values()), "red", label='Validation Loss')
    plt.legend()
    plt.show()

    # Plot graph for the accuracy scores.
    train_set_plot = plt.plot(list(train_set_accuracy_score.keys()), list(train_set_accuracy_score.values()), "black", label='Train Accuracy')
    validation_set_plot = plt.plot(list(validation_set_accuracy_score.keys()), list(validation_set_accuracy_score.values()), "red", label='Validation Accuracy')
    plt.legend()
    plt.show()


def predict_test_set(nn_model, test_x):
    correct_predictions_list = []

    nn_model.eval()
    for x in test_x:
        outputs = nn_model(x)
        max_arg_outputs = torch.max(outputs, 1)[1]
        correct_predictions_list.append(max_arg_outputs)

    return correct_predictions_list


def create_predictions_file(test_set_predicts, file_name):
    predictions_string = "\n".join(str(prediction.item()) for prediction in test_set_predicts)
    file = open(file_name, "w")
    file.write(predictions_string)
    file.close()


def main():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_mnist_loader = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_mnist_loader = datasets.FashionMNIST('./data', train=False, transform=transform)

    train_set_length = len(train_mnist_loader)
    threshold = int(0.2 * train_set_length)

    list_of_indices = list(range(train_set_length))
    validation_set_indices = np.random.choice(list_of_indices, size=threshold, replace=False)
    train_set_indices = list(set(list_of_indices) - set(validation_set_indices))

    validation_set_sampler = SubsetRandomSampler(validation_set_indices)
    train_set_sampler = SubsetRandomSampler(train_set_indices)

    train_set = torch.utils.data.DataLoader(train_mnist_loader, batch_size=MINI_BATCH_SIZE, sampler=train_set_sampler)
    validation_set = torch.utils.data.DataLoader(train_mnist_loader, batch_size=1, sampler=validation_set_sampler)
    test_set = torch.utils.data.DataLoader(test_mnist_loader, batch_size=MINI_BATCH_SIZE, shuffle=True)

    nn_model = NeuralNetworkC(784)
    train_neural_network(nn_model, train_set, validation_set)

    calculate_loss_and_accuracy(nn_model, test_set, MINI_BATCH_SIZE, "FashionMNIST Test Set")

    test_x = torch.Tensor(np.loadtxt("test_x") / 255)
    test_set_predicts = predict_test_set(nn_model, test_x)
    create_predictions_file(test_set_predicts, "test_y")


if __name__ == "__main__":
    main()
