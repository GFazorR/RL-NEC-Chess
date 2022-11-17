import numpy as np
from DifferentiableNeuralDictionary import DifferentiableNeuralDictionary as DND
from torch.nn.functional import smooth_l1_loss
from torch.autograd import Variable
import torch
import random
import math


class NeuralEpisodicControl:
    def __init__(self, env, q_network, replay_buffer, optimizer, n_steps=1, gamma=.99, eps_start=.3, eps_end=.05,
                 eps_decay=200, alpha=1e-3, batch_size=64, loss_fnc=smooth_l1_loss):
        self.env = env
        self.q_network = q_network
        self.replay_buffer = replay_buffer
        self.memory = {}
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

        self.kernel = torch.nn.CosineSimilarity(dim=0)
        # Threshold for similarity
        self.tau = 0.5

    # TODO modify method
    def select_action(self, state):
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        actions = self.env.get_legal_moves()
        if random.random() > eps:
            # Greedy choice
            idx = torch.argmax(self.attend(state, actions))
            return random.choice(actions[idx])
        else:
            # Random choice
            return random.choice(actions)

    def train(self, episodes):
        rewards = []
        for i in range(episodes):
            state = self.env.reset()
            rewards.append(self.play_episode(state))
        return rewards

    # TODO Self Play
    def play_episode(self, state):
        steps = 0
        cumulative_reward = 0
        while True:
            g_n = 0
            for n in range(self.n_steps):
                h = self.q_network(state)
                action = self.select_action(h)
                if n == 0:
                    first_action = action
                    first_state = h
                next_state, reward, done = self.env.step(action)
                cumulative_reward += reward
                steps += 1
                g_n += (self.gamma**n) * reward
                state = next_state
                if done:
                    break
            # Get legal moves
            actions = [self.env.get_legal_moves()]
            # Calculate G_n (Bellman Target)
            attention = self.attend(h, actions)
            tabular_q_value = g_n + (self.alpha ** n) * attention
            # If action not in memory create new DND
            
            if self.memory.get(first_action):
                self.memory[first_action] = DND(100, 128, 50)

            # If max similarity < threshold tau then write new value to DND
            if torch.max(self.kernel(self.memory[first_action].lookup(h), h)) < self.tau:
                self.memory[first_action].write(h, tabular_q_value)

            self.replay_buffer.enqueue(
                first_state,
                first_action,
                h,
                tabular_q_value
            )

            # Tabular update
            self.memory[first_action].update(self.alpha, g_n)

            self.learn()
            if done:
                break
        return cumulative_reward

    def get_transitions(self):
        # sample randomly from replay buffer
        transitions = self.replay_buffer.sample(self.batch_size)
        return [Variable(torch.cat(transition)) for transition in zip(*transitions)]

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch_state, batch_action, batch_next_state, batch_reward = self.get_transitions()
        q_values = torch.zeros(self.batch_size)
        for i,action in enumerate(batch_action):
            hs, values = self.memory[action].lookup(batch_next_state[i])
            # compute attention
            kernel_sum = self.kernel(hs, batch_state[i])
            w = kernel_sum / torch.sum(kernel_sum)
            q_values[i] = torch.sum(w * values)

        loss = self.loss_fnc(batch_reward, q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def attend(self, h, actions):
        q_values = torch.zeros(len(actions))
        for i, action in enumerate(actions):
            # generate key
            hs, values = self.memory[action].lookup(h)
            # compute attention
            kernel_sum = self.kernel(hs, h)
            w = kernel_sum / torch.sum(kernel_sum)
            q_values[i] = torch.sum(w * values)

        return q_values


if __name__ == '__main__':
    pass
