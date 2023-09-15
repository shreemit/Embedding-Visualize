# Training script for Fashion-MNIST dataset

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


# Training settings
parser = argparse.ArgumentParser(description='Fashion-MNIST Training Script')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='number of batches to wait before logging training status (default: 100)')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
parser.add_argument('--log-dir', type=str, default='logs',
                    help='Directory for Tensorboard logs')

args = parser.parse_args()

# Set up Tensorboard
writer = SummaryWriter(args.log_dir)

# Set up device
device = torch.device("mps" if torch.cuda.is_available() else "cpu")

# Set up data loaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                            shuffle=False)

# Set up model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = Net().to(device)

# Set up optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# Set up loss function
criterion = nn.CrossEntropyLoss()

# Training loop
def train(loader, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        writer.add_scalar('Loss/train', loss.item(), epoch*len(loader) + batch_idx)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx*len(data), len(loader.dataset),
                    100. * batch_idx / len(loader), loss.item()))

# Test loop
def test(loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', accuracy, epoch)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(loader.dataset), accuracy))

# Train and test
for epoch in range(1, args.epochs + 1):
    train(trainloader, epoch)
    test(testloader, epoch)

# Save model
if args.save_model:
    torch.save(model.state_dict(), "fashion_mnist_cnn.pt")

# Close Tensorboard writer
writer.close()
