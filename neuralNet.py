import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.lin1 = nn.Linear(in_features = 784, out_features = 500)
        self.relu = nn.LeakyReLU(negative_slope = 0.01)
        self.lin2 = nn.Linear(in_features = 500, out_features = 300)
        self.lin3 = nn.Linear(in_features = 300, out_features = 10)


    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)

        return x

