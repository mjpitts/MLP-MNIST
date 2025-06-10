import torch 
import torchvision
import matplotlib.pyplot as plt
from util import logMessage
from neuralNet import NeuralNet
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report

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

def plotExample(data, truth):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(truth[i]))
        plt.xticks([])
        plt.yticks([])
    plt.savefig('./figures/example_data.png')

def main():

    # Init our training hyperparameters
    n_epochs = 1
    train_batch_size = 100
    test_batch_size = 1000
    learning_rate = 1e-4

    # Set device to GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Init loader objects
    train_loader, test_loader = initLoaders(train_batch_size, test_batch_size)

    # Cast test_loader as iter and take first batch as an example
    test_loader = iter(test_loader)
    example_data, example_label = next(test_loader)

    # Print batch shape
    logMessage(f"Batch Shape: {example_data.shape}")

    # Plot examples and save to ./figures
    plotExample(example_data, example_label)

    # Init model
    model = NeuralNet().to(device)
    model.train()

    # Init loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

    # Array of loss history
    loss_history = []
    steps = [i for i in range((6000//train_batch_size) * n_epochs)]

    # Train the model
    n_total_steps = len(train_loader)
    for epoch in range(n_epochs):
        for i, (images, labels) in enumerate(train_loader):  
            # origin shape: [100, 1, 28, 28]
            # resized: [100, 784]
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            
            # Forward pass and loss calculation
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if(i+1) % 10 == 0:
                loss_history.append(loss.item())

            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

    #  Test the model: we don't need to compute gradients
    with torch.no_grad():
        model.eval()
        n_correct = 0
        # minus 1, because on batch was used to create figures.
        n_samples = (len(test_loader) - 1) * test_batch_size

        total_pred = []
        total_labs = []

        while True:
            try:
                images, labels = next(test_loader)
                images = images.reshape(-1, 28*28).to(device)
                labels = labels.to(device)
                # Accumulate labels
                for i in labels.cpu():
                    total_labs.append(int(i))

                outputs = model.forward(images)

                # max returns (output_value ,index)
                _, predicted = torch.max(outputs, 1)
                # Accumulate predictions
                for i in predicted.cpu():
                    total_pred.append(int(i))
                n_correct += (predicted == labels).sum().item()

            except StopIteration:
                acc = n_correct / n_samples
                print(f'Accuracy of the network on the {n_samples} test images: {100*acc} %')
                break


        # Create confusion matrix
        confusionMat = confusion_matrix(total_pred, total_labs, labels=[0,1,2,3,4,5,6,7,8,9])

        # Print class metrics    
        print(classification_report(total_pred, total_labs))
        
        # Init array that counts total true instances of this class
        class_count = [0 for i in range(10)]
        # Init array that counts total number of prediction of this 
        total_pred_count = [0 for i in range(10)]

        # Create class count matrix
        for num in range(10):
            for i, pred_count in enumerate(confusionMat[num]):
                class_count[i] += int(pred_count)
                total_pred_count[num] += int(pred_count)


        # Plot and save loss.
        plt.clf()
        plt.plot(steps, loss_history, label="loss")

        plt.xlabel('loss')
        plt.ylabel('step')

        plt.legend()

        plt.savefig('./figures/Loss_plot.png')

if __name__ == "__main__":
    main()