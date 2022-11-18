import numpy as np
from DifferentiableNeuralDictionary import DifferentiableNeuralDictionary
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

    def select_action(self, state):
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        actions = self.env.get_legal_moves()

        if random.random() > eps:
            # Greedy choice
            idx = torch.argmax(self.compute_attention(state, actions))

            return random.choice([a for i, a in enumerate(actions) if i in idx])
        else:
            # Random choice
            return random.choice(actions)

    def train(self, episodes):
        rewards = []
        for i in range(episodes):
            observation = self.env.reset()

            # Random Move if opponent plays
            opponent_starts = random.choice([True, False])
            print(opponent_starts)
            if opponent_starts:
                action = random.choice(self.env.get_legal_moves())
                observation, _, _ = self.env.step(action)

            rewards.append(self.play_episode(observation))
        return rewards

    def play_episode(self, observation):
        steps = 0
        cumulative_reward = 0
        state = self.q_network(observation)

        while True:
            # Play n steps
            first_state, first_action, state, g_n, n, done = \
                self.play_n_steps(state, cumulative_reward, steps)

            # Get legal moves
            actions = self.env.get_legal_moves()

            # Calculate G_n (Bellman Target)
            attention = torch.max(self.compute_attention(state, actions))
            tabular_q_value = g_n + (self.alpha ** n) * attention

            # If max similarity < threshold tau then write new value to DND
            # if torch.max(self.kernel(self.memory[first_action].lookup(state), state)) < self.tau:
            if not self.memory.get(first_action):
                self.memory[first_action] = DifferentiableNeuralDictionary(100, 128, 10)

            self.memory[first_action].write(state, tabular_q_value)

            # first_state: embedding
            # first_action: string
            # state: embedding
            # tabular_q_value: float
            self.replay_buffer.enqueue((first_state, first_action, state,
                                        self.float_tensor([tabular_q_value])))

            # Tabular update
            self.memory[first_action].update(self.alpha, g_n)

            self.learn()

            if done:
                break
        return cumulative_reward

    def get_transitions(self):
        # sample randomly from replay buffer
        transitions = self.replay_buffer.sample(self.batch_size)
        state, action, next_state, g_n = zip(*transitions)
        batch_state = Variable(torch.stack(state))
        next_state = Variable(torch.stack(next_state))
        estimated_q_vals = Variable(torch.cat(g_n))
        return batch_state, action, next_state, estimated_q_vals

    def play_n_steps(self, state, cumulative_reward, steps):
        g_n = 0
        for n in range(self.n_steps):
            # play agent turn
            action = self.select_action(state)

            # save first state/action
            if n == 0:
                first_action = action
                first_state = state
            # perform the action
            next_state, reward, done = self.env.step(action)

            # play opponent turn
            if not done:
                state = self.q_network(next_state)
                opponent_action = self.select_action(state)
                next_state, reward, done = self.env.step(opponent_action)

            # compute stats
            cumulative_reward += reward
            steps += 1

            # cumulative discounted reward
            g_n += (self.gamma ** n) * reward
            # update state
            state = self.q_network(next_state)

            if done:
                break

        return first_state, first_action, state, g_n, n, done

    def learn(self):
        # TODO add condition
        if len(self.replay_buffer) < self.batch_size:
            return
        batch_state, batch_action, batch_next_state, batch_reward = self.get_transitions()
        q_values = torch.zeros(self.batch_size)
        for i, action in enumerate(batch_action):
            if self.memory[action].is_queryable():
                hs, values = self.memory[action].lookup(batch_next_state[i].reshape(1, -1))
                hs = hs.reshape((10, 128))
                current_state = batch_state[i].reshape((1, 128))

                # compute attention
                kernel_sum = self.kernel(hs, current_state)
                w = kernel_sum / torch.sum(kernel_sum)

                q_values[i] = torch.sum(w * values)
            else:
                q_values[i] = 0.0

        loss = self.loss_fnc(Variable(batch_reward), Variable(q_values))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # TODO Refactor
    def compute_attention(self, state, actions):
        q_values = torch.zeros(len(actions))
        for i, action in enumerate(actions):
            # generate key
            # If action not in memory create new DND
            if self.memory.get(action) is None:
                self.memory[action] = DifferentiableNeuralDictionary(100, 128, 10)

            if self.memory[action].is_queryable():
                self.attend(state, action)

            else:
                q_values[i] = 0

        return q_values

    # TODO fix mess with shapes
    def attend(self, state, action):
        hs, values = self.memory[action].lookup(state)
        # compute attention
        kernel_sum = self.kernel(hs, state)
        w = kernel_sum / torch.sum(kernel_sum)
        return torch.sum(w * values)


if __name__ == '__main__':
    pass
