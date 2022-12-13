import numpy as np
from DifferentiableNeuralDictionary import DifferentiableNeuralDictionary
from torch.nn.functional import smooth_l1_loss
from torch.autograd import Variable
import torch
import random
import math
from tqdm import tqdm


class NeuralEpisodicControl:
    def __init__(self, env, q_network, replay_buffer, optimizer, n_steps=1, gamma=.99, eps_start=.3, eps_end=.01,
                 eps_decay=1000, alpha=0.5, batch_size=64, loss_fnc=smooth_l1_loss):
        self.env = env
        self.q_network = q_network
        self.replay_buffer = replay_buffer
        self.memory = {}
        self.optimizer = optimizer(self.q_network.parameters(), 1e-3)
        self.loss_fnc = loss_fnc

        self.n_steps = n_steps
        self.gamma = gamma
        self.alpha = alpha

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

        self.kernel = torch.nn.CosineSimilarity(dim=1)  # TODO change 
        # Threshold for similarity
        self.tau = 0.1

    def select_action(self, state, episode):
        # if # TODO check the decay
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * episode / self.eps_decay)
        actions = self.env.get_legal_moves()

        if random.random() > eps:
            # Greedy choice
            idx = torch.argmax(self.compute_attention(state, actions))

            return random.choice([a for i, a in enumerate(actions) if i in idx])
        else:
            # Random choice
            return random.choice(actions)
    def opponent_action(self,state, episode):
         # if # TODO check the decay
        eps = self.eps_end + (.9 - self.eps_end) * math.exp(-1. * episode / self.eps_decay)
        actions = self.env.get_legal_moves()

        if episode < 500:
            return random.choice(actions)
        else:
            if random.random() > eps:
                # Greedy choice
                idx = torch.argmax(self.compute_attention(state, actions))

                return random.choice([a for i, a in enumerate(actions) if i in idx])
            else:
                # Random choice
                return random.choice(actions)

    def train(self, episodes):
        rewards = []
        
        for i in tqdm(range(episodes), total=episodes):
            observation = self.env.reset()

            # Random Move if opponent plays
            opponent_starts = random.choice([True, False])
            if opponent_starts:
                action = random.choice(self.env.get_legal_moves())
                observation, _, _ = self.env.step(action, opponent_starts)

            rewards.append(self.play_episode(observation, opponent_starts, i))
            
        return rewards

    def play_episode(self, observation, opponent_starts, episode):
        steps = 0
        cumulative_reward = 0
        state = self.q_network(observation)

        while True:
            # Play n steps
            first_state, first_action, state, g_n, n, done, c_r = \
                self.play_n_steps(state, cumulative_reward, opponent_starts, episode)
            steps += n
            cumulative_reward += c_r

            # Get legal moves
            actions = self.env.get_legal_moves()

            # Calculate G_n (Bellman Target)
            if len(actions) > 0:
                attention = torch.max(self.compute_attention(state, actions))
            else:
                attention = torch.zeros(1)
            tabular_q_value = g_n + (self.alpha ** n) * attention

            # If max similarity < threshold tau then write new value to DND
            if not self.memory.get(first_action):
                self.memory[first_action] = DifferentiableNeuralDictionary(100, 64, 10)

            if self.memory[first_action].is_queryable():
                if torch.max(self.kernel(self.memory[first_action].lookup(state)[0], state)) < self.tau:
                    self.memory[first_action].write(state, tabular_q_value)
            else:
                self.memory[first_action].write(state, tabular_q_value)

            # first_state: embedding
            # first_action: string
            # state: embedding
            # tabular_q_value: float
            self.replay_buffer.enqueue((first_state, first_action, state,
                                        self.float_tensor([tabular_q_value])))

            # Tabular update
            self.memory[first_action].update(self.gamma, g_n, first_state, self.kernel)

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

    def play_n_steps(self, state, cumulative_reward, opponent_starts, episode):
        g_n = 0
        for n in range(self.n_steps):
            # play agent turn
            action = self.select_action(state, episode)

            # save first state/action
            if n == 0:
                first_action = action
                first_state = state
            # perform the action
            next_state, reward, done = self.env.step(action, not opponent_starts)
            if reward == 1.:
                print(reward)

            # play opponent turn
            if not done:
                # state = self.q_network(next_state)
                actions = self.env.get_legal_moves()
                opponent_action = self.opponent_action(next_state, episode)  # next state
                next_state, reward, done = self.env.step(opponent_action, opponent_starts)
                if reward == 1.:

                    reward = -reward
                    print(reward)
                

            # compute stats
            cumulative_reward += reward

            # cumulative discounted reward
            g_n += (self.alpha ** n) * reward
            # update state
            state = self.q_network(next_state)

            if done:
                break

        return first_state, first_action, state, g_n, n, done, cumulative_reward

    def learn(self):
        # TODO add condition
        if len(self.replay_buffer) < self.batch_size:
            return
        batch_state, batch_action, batch_next_state, batch_reward = self.get_transitions()
        q_values = torch.zeros(self.batch_size)
        for i, action in enumerate(batch_action):
            if self.memory[action].is_queryable():
                hs, values = self.memory[action].lookup(batch_next_state[i])
                # compute attention
                kernel_sum = self.kernel(hs, batch_state[i])
                w = kernel_sum / torch.sum(kernel_sum)

                q_values[i] = torch.sum(w * values)
            else:
                q_values[i] = 0.0

        q_values.requires_grad = True

        loss = self.loss_fnc(batch_reward, q_values)

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
                self.memory[action] = DifferentiableNeuralDictionary(100, 64, 10)

            if self.memory[action].is_queryable():
                q_values[i] = self.attend(state, action)

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
