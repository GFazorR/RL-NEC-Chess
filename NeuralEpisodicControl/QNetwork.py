from torch import nn
from torchsummary import summary
import torch
from Chess import Chess


class QNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding='same')
        self.conv_2 = nn.Conv2d(in_channels=36, out_channels=72, kernel_size=3, padding='same')
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        self.max_1 = nn.MaxPool2d(2,2)
        self.global_avg = nn.AvgPool2d(3,3)

        self.flatten = nn.Flatten()

    def forward(self, x):
        inputs = x
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = torch.cat((x, inputs), dim=1)
        x = self.max_1(x)
        skip_layer_1 = x
        x = self.conv_2(x)
        x = self.relu_2(x)
        x = torch.cat((x, skip_layer_1), dim=1)
        x = self.global_avg(x)
        x = self.flatten(x)
        return x.reshape(-1, 108)


if __name__ == '__main__':
    # model = QNetwork()
    # summary(model, (12,8,8))
    # env = Chess()
    # state, reward, done = env.step('e4')
    # out = model(state)
    # print(out.shape)
    pass
