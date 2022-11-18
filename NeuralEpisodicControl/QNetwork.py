from torch import nn
from torch.nn.functional import relu


class QNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool(relu(self.conv1(x)))
        x = self.flatten(relu(self.conv2(x)))
        return x.reshape(128)


if __name__ == '__main__':
    pass
