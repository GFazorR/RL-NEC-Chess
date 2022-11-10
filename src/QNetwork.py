from torch import nn
if __name__ == '__main__':
    pass


class QNetwork(nn.Module):
    def __init__(self, input_layer, hidden_layer, output_layer) -> None:
        super().__init__()
        self.relu_stack = nn.Sequential(
            nn.Linear(input_layer, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, output_layer)
        )

    def forward(self, x):
        return self.relu_stack(x)
