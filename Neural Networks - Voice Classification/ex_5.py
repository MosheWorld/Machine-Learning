import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch import cuda
from datetime import datetime
from gcommand_loader import GCommandLoader

num_epochs = 8
batch_size = 64
learning_rate = 0.0007
image_size = 20 * 11 * 100


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.cnn_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.cnn_block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.cnn_block4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.mlp = nn.Sequential(nn.Linear(1920, 500),
                                 nn.Dropout(0.5),
                                 nn.ReLU(),
                                 nn.Linear(500, 100),
                                 nn.Dropout(0.5),
                                 nn.ReLU(),
                                 nn.Linear(100, 30))

    def forward(self, x):
        out = self.cnn_block1(x)
        out = self.cnn_block2(out)
        out = self.cnn_block3(out)
        out = self.cnn_block4(out)
        out = out.reshape(out.size(0), -1)
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)


def save_graph(train, test, y_axis, title=None):
    plt.suptitle(y_axis, fontsize=20)
    plt.figure()
    plt.plot(train, color='r', label='train')
    plt.plot(test, color='g', label='validation')
    plt.xlabel('Epochs')
    plt.legend(loc="upper left")
    plt.ylabel(y_axis)
    plt.title(y_axis if not title else title)
    plt.savefig(y_axis+'.png')


def evaluate(model, data_loder, set_name):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loder):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += F.nll_loss(outputs, labels, reduction='sum')
            pred = outputs.max(1, keepdim=True)[1].view(-1)
            correct += (pred == labels).sum().item()
            total += pred.shape[0]
        test_loss /= len(data_loder.dataset)
        test_acc = 100. * correct / total
        print(f'{datetime.now()}: {set_name} set: '
              f'Accuracy: {correct}/{total}({test_acc:.2f}%), '
              f'Average loss: {test_loss:.8f}')
    return test_acc, test_loss


def train(model, optimizer, train_loader, device):
    loss_list = {'Train': [], 'Validation': []}
    acc_list = {'Train': [], 'Validation': []}
    print(f"{datetime.now()}: Start training phase")
    for epoch in range(num_epochs):
        print(f"{datetime.now()}: Epoch:", epoch+1)
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            # Backprop with Adam optimization
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            # Run the forward pass
            outputs = model(images)
            loss = F.nll_loss(outputs, labels)
            loss.backward()
            optimizer.step()

        acc, lss = evaluate(model, train_loader, "Train")
        acc_list['Train'].append(acc)
        loss_list['Train'].append(lss)
        acc, lss = evaluate(model, valid_loader, "Validation")
        acc_list['Validation'].append(acc)
        loss_list['Validation'].append(lss)

    save_graph(acc_list['Train'], acc_list['Validation'], 'Accuracy')
    save_graph(loss_list['Train'], loss_list['Validation'], 'Loss')


def print_test_y(input, model, device, data_test, train_loader):
    file = open("test_y", "w")
    for i, (images, labels) in enumerate(input):
        name = data_test.spects[i][0]
        name = name.split("\\")[-1]
        images = images.to(device)
        res = model(images)
        pred = res.argmax(1)
        string_to_return = name + "," + str(train_loader.dataset.classes[pred[0].item()]) + "\n"
        file.write(string_to_return)
        #break
    file.close()


if __name__ == "__main__":
    device = 'cuda' if cuda.is_available() else 'cpu'
    print("Graphical device test: {}".format(torch.cuda.is_available()))
    print("{} available".format(device))

    data_train = GCommandLoader("data/train")
    data_valid = GCommandLoader("data/valid")
    data_test = GCommandLoader("data/test")

    test_loader = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=False)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(data_valid, batch_size=128, shuffle=True)

    model = CNN().to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train(model, optimizer, train_loader, device)
    print_test_y(test_loader, model, device, data_test, train_loader)
