import torch
import torch.optim as optim
import pandas as pd
import numpy as np

import Model
import MusicData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

model = Model.CNN().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loader = MusicData.train_loader
test_loader = MusicData.test_loader

for epoch in range(10):
    model.train()
    for i, (data, label) in enumerate(train_loader):
        # print data shape
        # print(data.shape)

        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.detach().cpu().item()}")

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            proba, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            # if predicted == label:
            #     print(
            #         f"probabilities: {proba}, predicted: {predicted}, actual: {label}"
            #     )
        print(f"Epoch: {epoch}, Accuracy: {100 * correct / total}")
