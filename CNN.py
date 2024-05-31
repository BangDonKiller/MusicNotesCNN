import torch
import torch.optim as optim
import pandas as pd
import numpy as np

import Model
import MusicData
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


class MusicNote_CNN():

    def __init__(self, weights=None):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model.CNN().to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.train_loader = MusicData.train_loader
        self.test_loader = MusicData.test_loader
        self.test_one_loader = MusicData.test_one_loader

        if weights is not None:
            self.model.load_state_dict(torch.load(weights))
            print("loaded weights from: ", weights)

        print("training using device: ", self.device)

    def train(self,
              epochs=10,
              save_weights='model_weight/CNN_model_PixelNotes.pt'):
        for epoch in range(epochs):
            self.model.train()
            for i, (data, label) in enumerate(self.train_loader):
                data, label = data.to(self.device), label.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, label)

                loss.backward()
                self.optimizer.step()
                if i % 100 == 0:
                    print(
                        f"Epoch: {epoch}, Loss: {loss.detach().cpu().item()}")

            self.model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for data, label in self.test_loader:
                    data, label = data.to(self.device), label.to(self.device)
                    output = self.model(data)
                    proba, predicted = torch.max(output, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
                print(f"Epoch: {epoch}, Accuracy: {100 * correct / total}")

        torch.save(self.model.state_dict(), save_weights)

    def test_one_case(self):
        self.model.eval()
        with torch.no_grad():
            # test only one sample
            data, label = next(iter(self.test_loader))
            data, label = data.to(self.device), label.to(self.device)
            output = self.model(data)
            proba, predicted = torch.max(output, 1)

            print(
                f"probabilities: {output}, predicted: {predicted}, actual: {label}"
            )

            data, label = next(iter(self.test_one_loader))
            # 使用torchvision.transforms將Tensor轉換為PIL圖像
            transform = transforms.ToPILImage()
            image = transform(
                torch.tensor(np.array(data.cpu()).reshape(1, 64, 64)))
            image = image.convert('L')
            image = ImageOps.invert(image)

            transform_2 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((28, 28)),
            ])
            image2 = transform_2(
                torch.tensor(np.array(data.cpu()).reshape(1, 64, 64)))
            image2 = image2.convert('L')  # type: ignore
            image2 = ImageOps.invert(image2)

            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(image, cmap="gray")  # type: ignore
            ax[0].set_title("original")
            ax[1].imshow(image2, cmap="gray")  # type: ignore
            ax[1].set_title("scaled")
            plt.show()


def main():
    MODE = 2  # 1: train a new network, 2: test current network

    if MODE == 1:
        musicNote_CNN = MusicNote_CNN()
        musicNote_CNN.train(
            save_weights='model_weight/CNN_model_PixelNotes.pt')
    elif MODE == 2:
        musicNote_CNN = MusicNote_CNN(
            weights='model_weight/CNN_model_PixelNotes.pt')
        musicNote_CNN.test_one_case()


if __name__ == '__main__':
    main()
