from torch import nn
from torchsummary import summary
import torch
from Chess import Chess
from DifferentiableNeuralDictionary import DifferentiableNeuralDictionary


class QNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding='same')
        self.conv_2 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, padding='same')
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        self.max_1 = nn.MaxPool2d(2,2)
        self.global_avg = nn.AvgPool2d(3,3)
        self.flatten = nn.Flatten()
        self.memory = {}

    def forward(self, x, actions, learning=False):
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.max_1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.global_avg(x)
        state = self.flatten(x)
        state.reshape(-1, 48)
        # actions = [a for a in actions if a in self.memory and self.memory[a].is_queryable()]
        attention = self.compute_attention(state, actions, learning)

        if learning:
            q_values, ids, neighbours, values = zip(*attention)
            print()
            return torch.tensor(q_values), ids, torch.cat(neighbours,dim=1), torch.cat(values, dim=1)
        else:
            return state, torch.tensor(attention)

    def compute_attention(self, state, actions, learning):
        out = []
        for i, action in enumerate(actions):
            # generate key (already generated in previous steps)
            # If action not in memory create new DND
            if self.memory.get(action) is None:
                self.memory[action] = DifferentiableNeuralDictionary(100, 48)
            if learning:
                if self.memory[action].is_queryable():
                    out.append(self.memory[action](state, learning))
            else:
                if self.memory[action].is_queryable():
                    out.append(self.memory[action](state, learning))
                else:
                    out.append(torch.zeros(1))
        return out

if __name__ == '__main__':
    model = QNetwork()
    print(summary(model, (12,8,8)))
    env = Chess()
    state, reward, done = env.step('e4')
    actions = env.get_legal_moves()
    out = model(state,actions,False)
    print(out)
    pass
