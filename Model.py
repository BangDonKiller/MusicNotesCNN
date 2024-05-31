import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 50, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(50, 100, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(100, 200, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classfier = nn.Sequential(nn.Linear(200 * 1 * 1, 100), nn.ReLU(),
                                       nn.Linear(100, 50), nn.ReLU(),
                                       nn.Linear(50, 5))

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 200 * 1 * 1)
        x = self.classfier(x)
        return x
