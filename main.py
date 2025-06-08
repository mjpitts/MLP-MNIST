import torch 
import torchvision
import matplotlib.pyplot as plt
from util import logMessage

"""
input: Hyperparameters thsat will define the loader batch sizes, ie: how many
pictures are trained on before updating the gradient.
output: void

Initalizes the loader objects.
"""
def initLoaders(train_batch_size, test_batch_size):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./dataset', train=True, download=True,
            transform=torchvision.transforms.ToTensor()),
            batch_size=train_batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./dataset', train=False, download=True,
            transform=torchvision.transforms.ToTensor()),
            batch_size=test_batch_size, shuffle=True)

    logMessage("Loaders Initialized")    

    return train_loader, test_loader



def main():

    # Init our training hyperparameters
    n_epochs = 3
    train_batch_size = 64
    test_batch_size = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    # init loader objects
    train_loader, test_loader = initLoaders(train_batch_size, test_batch_size)

if __name__ == "__main__":
    main()