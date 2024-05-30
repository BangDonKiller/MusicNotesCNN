from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torchvision import transforms
import torch


class MusicDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.labels = {
            label: idx
            for idx, label in enumerate(sorted(self.data['label'].unique()))
        }
        self.img_data = self.data.drop(columns=['label']).values.astype(
            np.float32)
        self.img_labels = self.data['label'].map(self.labels).values.astype(
            np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if (self.img_data[idx].shape[0] == 784):
            img = self.img_data[idx].reshape((28, 28))
        else:
            img = self.img_data[idx].reshape((64, 64))
        label = self.img_labels[idx]

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        return img, label


# 確保 transform 處理正確
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# loader 版本
train_loader = DataLoader(MusicDataset('./dataset.csv', transform=transform),
                          batch_size=10,
                          shuffle=True)
test_loader = DataLoader(MusicDataset('./data1.csv', transform=transform),
                         batch_size=10,
                         shuffle=True)
test_one_loader = DataLoader(MusicDataset('./data1.csv'),
                             batch_size=1,
                             shuffle=True)
