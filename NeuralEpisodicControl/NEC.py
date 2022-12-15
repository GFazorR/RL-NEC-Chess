import copy

import numpy as np
from DifferentiableNeuralDictionary import DifferentiableNeuralDictionary
from torch.nn.functional import smooth_l1_loss
from torch.autograd import Variable
import torch
import random
import math
from tqdm import tqdm
from Chess import Chess
from QNetwork import QNetwork
from ReplayBuffer import ReplayBuffer
import matplotlib.pyplot as plt


class NeuralEpisodicControl:
    def __init__(self, env, q_network, replay_buffer, optimizer, n_steps=1, gamma=.99, eps_start=.3, eps_end=.01,
                 eps_decay=200, alpha=0.5, batch_size=64, loss_fnc=smooth_l1_loss):
        self.env = env
        self.q_network = q_network
        self.replay_buffer = replay_buffer
        self.memory = {}
        self.optimizer = optimizer(self.q_network.parameters(), 1e-3)
        self.loss_fnc = loss_fnc

        self.n_steps = n_steps
        self.gamma = gamma
        self.alpha = alpha
        self.dnd_size = 500

        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_end = eps_end

        self.batch_size = batch_size

        use_cuda = torch.cuda.is_available()

        # TODO add LOG use_cuda
        self.float_tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.long_tensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

        if use_cuda:
            self.q_network.cuda()
        # self.opponent_network = copy.deepcopy(self.q_network)
        # self.opponent_memory = copy.deepcopy(self.memory)

        # Threshold for similarity
        self.tau = 0.6

    # =================== Training ===========================
    def train(self, episodes):
        rewards = []

        for i in tqdm(range(episodes), total=episodes):
            observation = self.env.reset()
            # opponent_starts = random.choice([True, False])
            # if opponent_starts:
            #     # Random Move if opponent plays first
            #     action = random.choice(self.env.get_legal_moves())
            #     observation, _, _ = self.env.step(action)

            rewards.append(self.play_episode(observation, i))
        return rewards

    def play_episode(self, state, episode):
        steps = 0
        cumulative_reward = 0

        # encode state

        while True:
            # Play n steps
            first_state, first_action, state, g_n, n, done, c_r = \
                self.play_n_steps(state, cumulative_reward, episode)
            steps += n
            cumulative_reward += c_r

            # Get legal moves
            actions = self.env.get_legal_moves()
            with torch.no_grad():
                embedded_state = self.q_network(torch.unsqueeze(state,0))
            # Calculate G_n (Bellman Target)
            if len(actions) > 0:
                attention = torch.max(self.compute_attention(embedded_state, actions, self.memory))
            else:
                attention = torch.zeros(1)

            tabular_q_value = g_n + (self.alpha ** n) * attention

            # If action not in dictionary, create new key with DND
            if not self.memory.get(first_action):
                self.memory[first_action] = DifferentiableNeuralDictionary(self.dnd_size, 108)

            # If max similarity < threshold tau then write new value to DND
            if self.memory[first_action].is_queryable():
                if self.memory[first_action].get_closest_distance(embedded_state.numpy()) < self.tau:
                    self.memory[first_action].write(embedded_state.numpy(), tabular_q_value)
            else:
                self.memory[first_action].write(embedded_state.numpy(), tabular_q_value)

            self.replay_buffer.enqueue((first_state, first_action, state,
                                        self.float_tensor([tabular_q_value])))

            # Tabular update
            with torch.no_grad():
                first_state = self.q_network(torch.unsqueeze(first_state,0))
            self.memory[first_action].update(self.gamma, tabular_q_value.numpy(), first_state.numpy())

            self.learn()

            if done:
                break

        return cumulative_reward

    def play_n_steps(self, state, cumulative_reward, episode):
        g_n = 0
        for n in range(self.n_steps):
            # play agent turn
            # select action
            action = self.select_action(state, episode, self.q_network, self.memory)

            # save first state/action
            if n == 0:
                first_action = action
                first_state = state

            # perform the action
            next_state, reward, done = self.env.step(action)

            # play opponent turn if game is not over
            if not done:
                # encode state

                # select action
                opponent_action = self.opponent_action(next_state, episode)
                # perform action
                next_state, reward, done = self.env.step(opponent_action)
                # Invert the reward
                if reward == 1:
                    reward = -reward

            # stats
            cumulative_reward += reward

            # cumulative discounted reward
            g_n += (self.gamma ** n) * reward
            # update state
            state = next_state

            if done:
                break

        return first_state, first_action, state, g_n, n, done, cumulative_reward

    def learn(self):

        if len(self.replay_buffer) < self.batch_size:
            return

        batch_state, batch_action, batch_next_state, batch_reward = self.get_transitions()
        q_values = torch.zeros(self.batch_size)
        batch_next_state = self.q_network(batch_next_state)
        for i, action in enumerate(batch_action):
            if self.memory[action].is_queryable():
                q_values[i] = self.memory[action] \
                    .attend(batch_next_state[i].detach().view(1, -1).numpy())
            else:
                q_values[i] = 0.0

        q_values.requires_grad = True

        calculated_loss = self.loss_fnc(batch_reward, q_values)

        self.optimizer.zero_grad()
        calculated_loss.backward()
        self.optimizer.step()

    # =================== Utility Functions ===================
    def select_action(self, state, episode, model, memory):
        with torch.no_grad():
            state = model(torch.unsqueeze(state,0))
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * episode / self.eps_decay)
        actions = self.env.get_legal_moves()

        if random.random() > eps:
            # Greedy choice
            idx = torch.argmin(self.compute_attention(state, actions, memory))
            return random.choice([a for i, a in enumerate(actions) if i in idx])
        else:
            # Random choice
            return random.choice(actions)

    def opponent_action(self, state, episode):
        with torch.no_grad():
            state = self.q_network(torch.unsqueeze(state,0))
        # eps = self.eps_end + (.7 - self.eps_end) * math.exp(-1. * episode / self.eps_decay)
        actions = self.env.get_legal_moves()
        if random.random() > .4:
            # Greedy choice
            idx = torch.argmax(self.compute_attention(state, actions, self.memory))

            return random.choice([a for i, a in enumerate(actions) if i in idx])
        else:
            # Random choice
            return random.choice(actions)

    def get_transitions(self):
        # sample randomly from replay buffer
        transitions = self.replay_buffer.sample(self.batch_size)
        state, action, next_state, g_n = zip(*transitions)
        batch_state = Variable(torch.stack(state))
        next_state = Variable(torch.stack(next_state))
        estimated_q_vals = Variable(torch.cat(g_n))
        return batch_state, action, next_state, estimated_q_vals

    # TODO Refactor
    def compute_attention(self, state, actions, memory):
        q_values = torch.zeros(len(actions))
        for i, action in enumerate(actions):
            # generate key (already generated in previous steps)
            # If action not in memory create new DND
            if memory.get(action) is None:
                memory[action] = DifferentiableNeuralDictionary(self.dnd_size, 108)

            if memory[action].is_queryable():
                q_values[i] = memory[action].attend(state.numpy())
            else:
                q_values[i] = 0

        return q_values


if __name__ == '__main__':
    agent = NeuralEpisodicControl(
        Chess(),
        QNetwork(),
        ReplayBuffer(50000),
        torch.optim.Adam,
        n_steps=5
    )
    results = agent.train(500)
    wins = results.count(1)
    loss = results.count(-1)
    draws = results.count(0)

    print(f'Wins: {wins}, Losses: {loss}, Draws: {draws}')

    plt.plot(results)
    plt.show()
