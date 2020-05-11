import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

epochs = 30
test_batch_size = 10
train_batch_size = 64
learning_rate = 0.001

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(root="./torch_datasets", train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root="./torch_datasets", train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)


class Autoencoder(nn.Module):
    def __init__(self, image_size):
        super(Autoencoder, self).__init__()
        self.image_size = image_size
        self.encoder_block1 = nn.Sequential(nn.Linear(in_features=self.image_size, out_features=512),nn.ReLU())
        self.encoder_block2 = nn.Sequential(nn.Linear(in_features=512, out_features=128),nn.ReLU())
        self.encoder_output = nn.Sequential(nn.Linear(in_features=128, out_features=128),nn.ReLU())

        self.decoder_block1 = nn.Sequential(nn.Linear(in_features=128, out_features=128),nn.ReLU())
        self.decoder_block2 = nn.Sequential(nn.Linear(in_features=128, out_features=512),nn.ReLU())
        self.decoder_output = nn.Sequential(nn.Linear(in_features=512, out_features=self.image_size),nn.ReLU())

    def forward(self, features):
        out = self.encoder_block1(features)
        out = self.encoder_block2(out)
        out = self.encoder_output(out)

        out = self.decoder_block1(out)
        out = self.decoder_block2(out)
        out = self.decoder_output(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder(image_size=784).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.MSELoss()

for epoch in range(epochs):
    loss = 0
    for batch_features, _ in train_loader:
        batch_features = batch_features.view(-1, 784).to(device)
        optimizer.zero_grad()
        outputs = model(batch_features)
        train_loss = loss_function(outputs, batch_features)
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()

    loss = loss / len(train_loader)
    print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))


test_examples = None
with torch.no_grad():
    for batch_features in test_loader:
        batch_features = batch_features[0]
        test_examples = batch_features.view(-1, 784).to(device)
        reconstruction = model(test_examples)
        break


with torch.no_grad():
    number = 10
    plt.figure(figsize=(20, 4))
    for index in range(number):
        # display original
        ax = plt.subplot(2, number, index + 1)
        plt.imshow(test_examples[index].cpu().numpy().reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, number, index + 1 + number)
        plt.imshow(reconstruction[index].cpu().numpy().reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()