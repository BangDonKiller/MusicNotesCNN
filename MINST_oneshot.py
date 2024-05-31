import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import pandas as pd
import numpy as np
from torchvision import transforms


class SiameseMNISTDataset(Dataset):

    def __init__(self, root, train=True, transform=None):
        self.mnist = datasets.MNIST(root=root,
                                    train=train,
                                    download=True,
                                    transform=transform)
        self.transform = transform
        self.train = train

        # 只保留每個類別的兩張圖片
        if self.train:
            self.data = []
            self.targets = []
            class_count = {}
            for img, label in self.mnist:
                if label not in class_count:
                    class_count[label] = 0
                if class_count[label] < 1:
                    self.data.append(img)
                    self.targets.append(label)
                    class_count[label] += 1
                if len(class_count) == 10 and all(
                        count == 2 for count in class_count.values()):
                    break

    def __getitem__(self, index):
        if self.train:
            img0, label0 = self.data[index], self.targets[index]
        else:
            img0, label0 = self.mnist[index]

        # 確保生成的隨機索引是整數型
        should_get_same_class = torch.randint(0, 2, (1, ),
                                              dtype=torch.int).item()
        if should_get_same_class:
            while True:
                if self.train:
                    idx = int(
                        torch.randint(0,
                                      len(self.data), (1, ),
                                      dtype=torch.int).item())
                    img1, label1 = self.data[idx], self.targets[idx]
                else:
                    idx = int(
                        torch.randint(0,
                                      len(self.mnist), (1, ),
                                      dtype=torch.int).item())
                    img1, label1 = self.mnist[idx]

                if label0 == label1:
                    break
        else:
            while True:
                if self.train:
                    idx = int(
                        torch.randint(0,
                                      len(self.data), (1, ),
                                      dtype=torch.int).item())
                    img1, label1 = self.data[idx], self.targets[idx]
                else:
                    idx = int(
                        torch.randint(0,
                                      len(self.mnist), (1, ),
                                      dtype=torch.int).item())
                    img1, label1 = self.mnist[idx]

                if label0 != label1:
                    break

        return (img0, img1), torch.tensor([int(label0 == label1)],
                                          dtype=torch.float32)

    def __len__(self):
        if self.train:
            return len(self.data)
        return len(self.mnist)


# 確保 transform 處理正確
transform = transforms.Compose(
    [transforms.Resize((28, 28)),
     transforms.ToTensor()])

# loader 版本
train_loader = DataLoader(SiameseMNISTDataset(root='./data',
                                              train=True,
                                              transform=transform),
                          batch_size=1,
                          shuffle=True)  # 調整batch_size
test_loader = DataLoader(SiameseMNISTDataset(root='./data',
                                             train=False,
                                             transform=transform),
                         batch_size=100,
                         shuffle=True)
test_one_loader = DataLoader(SiameseMNISTDataset(root='./data',
                                                 train=False,
                                                 transform=transform),
                             batch_size=1,
                             shuffle=True)
