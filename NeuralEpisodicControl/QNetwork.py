from torch import nn
from torch.nn.functional import relu
from torchsummary import summary
from Chess import Chess


class QNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool_1 = nn.MaxPool2d(2, 2)
        self.pool_2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool_1(relu(self.conv1(x)))
        x = self.flatten(self.pool_2(relu(self.conv2(x))))
        return x.reshape(-1, 64)
        


if __name__ == '__main__':
    pass
