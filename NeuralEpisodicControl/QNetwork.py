from torch import nn
from torchsummary import summary
from Chess import Chess


class QNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(3, 3),
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        return x.reshape(-1,24)


if __name__ == '__main__':
    # model = QNetwork()
    # summary(model, (6,8,8))
    # env = Chess()
    # state, reward, done = env.step('e4', False)
    # out = model(state)
    # print(out.shape)
    pass
