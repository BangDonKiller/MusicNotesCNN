import torch
import torch.optim as optim
import numpy as np
import Model
import MINST_oneshot
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import ImageOps


class MINST_Siamese:

    def __init__(self, pretrained_model=None, load_weights=None):
        '''
        ## Parameters:
        pretrained_model: the pretrained model path
        load_weights: the weights path

        training the model using the pretrained model and save the weights
        testing only load the weights and test one case, don't need pretrained model
        '''

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model_feature_extractor = Model.CNN()
        self.model = Model.SNN(self.model_feature_extractor.features).to(
            self.device)
        self.criterion = torch.nn.BCELoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.train_loader = MINST_oneshot.train_loader
        self.test_loader = MINST_oneshot.test_loader
        self.test_one_loader = MINST_oneshot.test_one_loader

        assert pretrained_model is not None or load_weights is not None, "pretrained_weights or weights should be provided"
        if pretrained_model is not None:
            self.model_feature_extractor.load_state_dict(
                torch.load(pretrained_model))
            print("Loaded pretrained model from: ", pretrained_model)

        # # Freeze the pretrained cnn model
        # for param in self.model_feature_extractor.parameters():
        #     param.requires_grad = False

        if load_weights is not None:
            self.model.load_state_dict(torch.load(load_weights))
            print("Loaded weights from: ", load_weights)

        print("Training using device: ", self.device)

    def train(self,
              epochs=1000,
              save_weights='model_weight/SNN_model_MINST.pt'):
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for i, ((img0, img1), label) in enumerate(self.train_loader):
                img0, img1, label = img0.to(self.device), img1.to(
                    self.device), label.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(img0, img1)
                loss = self.criterion(output, label)

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 10 == 9:
                    print(f"Epoch: {epoch}, Loss: {running_loss / 100:.3f}")
                    running_loss = 0.0

            if (epoch % 100 == 99):
                self.model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for (img0, img1), label in self.test_loader:
                        img0, img1, label = img0.to(self.device), img1.to(
                            self.device), label.to(self.device)
                        output = self.model(img0, img1)
                        proba = output > 0.5
                        total += label.size(0)
                        correct += (proba == label).sum().item()
                    print(
                        f"Epoch: {epoch}, Accuracy: {100 * correct / total:.2f}"
                    )

        torch.save(self.model.state_dict(), save_weights)

    def test_one_case(self):
        self.model.eval()
        with torch.no_grad():
            # Test only one sample
            (img0, img1), label = next(iter(self.test_one_loader))
            img0, img1, label = img0.to(self.device), img1.to(
                self.device), label.to(self.device)
            output = self.model(img0, img1)
            proba = output > 0.5

            print(
                f"Probability: {output.item()}, Predicted: {proba.item()}, Actual: {label.item()}"
            )

            # 使用torchvision.transforms將Tensor轉換為PIL圖像
            transform = transforms.ToPILImage()
            image0 = transform(
                torch.tensor(np.array(img0.cpu()).reshape(1, 28, 28)))
            image0 = image0.convert('L')
            image0 = ImageOps.invert(image0)

            image1 = transform(
                torch.tensor(np.array(img1.cpu()).reshape(1, 28, 28)))
            image1 = image1.convert('L')
            image1 = ImageOps.invert(image1)

            plt.subplot(1, 2, 1)
            plt.imshow(image0, cmap="gray")  # type: ignore
            plt.title("Image 0")

            plt.subplot(1, 2, 2)
            plt.imshow(image1, cmap="gray")  # type: ignore
            plt.title("Image 1")

            plt.show()


def main():
    MODE = 1  # 1: train a new network, 2: test current network

    if MODE == 1:
        minst_siamese = MINST_Siamese(
            pretrained_model='model_weight/CNN_model_PixelNotes.pt')
        minst_siamese.train(
            save_weights='model_weight/SNN_model_MINST_oneshot.pt')
    elif MODE == 2:
        minst_siamese = MINST_Siamese(
            load_weights='model_weight/SNN_model_MINST_oneshot.pt')
        minst_siamese.test_one_case()


if __name__ == '__main__':
    main()
