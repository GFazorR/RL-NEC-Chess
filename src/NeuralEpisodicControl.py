from torch.nn.functional import smooth_l1_loss
from torch.autograd import Variable
import torch
import random
import math

if __name__ == '__main__':
    pass


class NeuralEpisodicControl:
    def __init__(self, env, q_network, replay_buffer, dnd, optimizer, n_steps=1, gamma=.99, eps_start=.3, eps_end=.05,
                 eps_decay=200, alpha=1e-3, batch_size=64, loss_fnc=smooth_l1_loss):
        self.env = env
        self.q_network = q_network
        self.replay_buffer = replay_buffer
        self.dnd = dnd
        self.optimizer = optimizer(self.q_network.parameters(), alpha)
        self.loss_fnc = loss_fnc

        self.n_steps = n_steps
        self.gamma = gamma
        self.alpha = alpha

        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_end = eps_end

        self.batch_size = batch_size

        self.steps_done = 0

        use_cuda = torch.cuda.is_available()

        # TODO add LOG use_cuda
        self.float_tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.long_tensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

        if use_cuda:
            self.q_network.cuda()

    def select_action(self, state):
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if random.random() > eps:
            # Greedy choice
            return self.q_network(Variable(self.float_tensor(state))).data.max(1)[1].view(1, 1)
        else:
            # Random choice
            return self.long_tensor([[random.randrange(len(self.action_space))]])

    # TODO
    def train(self, episodes):
        pass

    # TODO
    def play_episode(self):
        pass

    def get_transitions(self):
        # sample randomly from replay buffer
        transitions = self.replay_buffer.sample(self.batch_size)
        return [Variable(torch.cat(transition)) for transition in zip(*transitions)]




