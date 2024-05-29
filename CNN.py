import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np


df = pd.read_csv('./dataset.csv')
train_data = df.drop(columns=['label']).values
train_data_label = df['label'].values

# 計算Train資料中的音符類別及數量
unique, counts = np.unique(train_data_label, return_counts=True)
print("Train dataset note amount: ", dict(zip(unique, counts)))

df2 = pd.read_csv('./data1.csv')
test_data = df2.drop(columns=['label']).values
test_data_label = df2['label'].values

# 計算Test資料中的音符類別及數量
unique, counts = np.unique(test_data_label, return_counts=True)
print("Test dataset note amount: ", dict(zip(unique, counts)))

def one_hot_encode(labels):



print("Before one-hot encoding: ", train_data_label)
print("After one-hot encoding: ", one_hot_encode(train_data_label))

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(28, 50, 3)
        self.conv2 = nn.Conv2d(50, 100, 3)
        self.conv3 = nn.Conv2d(100, 200, 3)
        self.fc1 = nn.Linear(200 * 1 * 1, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 12)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 200 * 1 * 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)