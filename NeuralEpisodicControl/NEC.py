from DifferentiableNeuralDictionary import DifferentiableNeuralDictionary
from torch.nn.functional import smooth_l1_loss
from torch.autograd import Variable
import torch
import random
import math
from tqdm import tqdm
from stockfish import Stockfish


class NeuralEpisodicControl:
    def __init__(self, env, q_network, replay_buffer, optimizer, n_steps=1, gamma=.90, eps_start=.3, eps_end=.01,
                 eps_decay=2000, alpha=0.5, batch_size=64, loss_fnc=smooth_l1_loss, opponent_start=.3,
                 opponent_end=.15):
        self.steps = 0
        self.env = env
        self.q_network = q_network
        self.replay_buffer = replay_buffer
        self.memory = {}
        self.optimizer = optimizer(self.q_network.parameters(), 1e-3)
        self.loss_fnc = loss_fnc

        self.n_steps = n_steps
        self.gamma = gamma
        self.alpha = alpha
        self.dnd_size = 100

        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_end = eps_end
        self.opponent_start = opponent_start
        self.opponent_end = opponent_end

        self.batch_size = batch_size

        use_cuda = torch.cuda.is_available()

        # TODO add LOG use_cuda
        self.float_tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.long_tensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

        if use_cuda:
            self.q_network.cuda()

        # Threshold for similarity
        self.tau = 0.7
        self.stockfish = Stockfish(
            path="C:\\Users\\coman\\Desktop\\stockfish_15.1_win_x64_avx2\\stockfish-windows-2022-x86-64-avx2.exe")
        self.stockfish.update_engine_parameters({
            "Contempt": 0,
            "Min Split Depth": 0,
            "Threads": 8,
            "Hash": 16,
            "Skill Level": 1,
            "Slow Mover": 100,
            "UCI_Chess960": "false",
            "UCI_LimitStrength": "false",
        })
        self.stockfish.set_depth(5)

    # =================== Training ===========================
    def train(self, episodes):
        rewards = []
        self.steps = 0
        for i in tqdm(range(episodes), total=episodes):
            observation = self.env.reset()
            # opponent_starts = random.choice([True, False])
            # if opponent_starts:
            #     # Random Move if opponent plays first
            #     action = random.choice(self.env.get_legal_moves())
            #     observation, _, _ = self.env.step(action)
            rewards.append(self.play_episode(observation))
            # self.play_episode(observation)
            # if i % 10 == 0:
            #     rewards.append(self.evaluate())
        return rewards

    def evaluate(self):
        c_r = 0
        state = self.env.reset()
        while True:
            action = self.select_action(state, True)
            next_state, reward, done = self.env.step(action)
            if not done:
                # opponent_action = random.choice(self.env.get_legal_moves())
                self.stockfish.set_fen_position(self.env.get_epd())
                opponent_action = self.stockfish.get_best_move()
                next_state, reward, done = self.env.step(opponent_action)
                if reward == 1:
                    reward = -reward
            c_r += reward
            state = next_state
            if done:
                if c_r == 1:
                    print("its a win!")
                elif c_r == 0:
                    print("at least its a draw!")
                break
        return c_r

    def play_episode(self, state):
        steps = 0
        cumulative_reward = 0

        # encode state

        while True:
            # Play n steps
            first_state, first_action, state, g_n, n, done, c_r = \
                self.play_n_steps(state, cumulative_reward)

            steps += n
            cumulative_reward += c_r


            # Calculate G_n (Bellman Target)
            actions = self.env.get_legal_moves()
            with torch.no_grad():
                embedded_state, q_values = self.q_network(torch.unsqueeze(state, 0), actions)
            if done:
                q_values = torch.zeros(1)

            tabular_q_value = g_n + (self.alpha ** n) * torch.max(q_values)

            # If max similarity < threshold tau then write new value to DND
            if self.q_network.memory.get(first_action) is None:
                self.q_network.memory[first_action] = DifferentiableNeuralDictionary(100, 48)

            if self.q_network.memory[first_action].is_queryable():
                most_similar = self.q_network.memory[first_action].get_closest_distance(embedded_state.numpy())
                if most_similar < self.tau:
                    self.q_network.memory[first_action].write(embedded_state.numpy(), tabular_q_value)
            else:
                self.q_network.memory[first_action].write(embedded_state.numpy(), tabular_q_value)

            self.replay_buffer.enqueue((first_state, first_action,
                                        self.float_tensor([tabular_q_value])))

            # Tabular update
            with torch.no_grad():
                first_state, _ = self.q_network(torch.unsqueeze(first_state, 0), [first_action], False)

            if self.q_network.memory[first_action].is_queryable():
                self.q_network.memory[first_action].update(self.gamma, tabular_q_value.numpy(), first_state.numpy())

            self.learn()

            if done:
                break
        if c_r == 1:
            print("its a win!")
        elif c_r == 0:
            print("at least its a draw!")
        return cumulative_reward

    def play_n_steps(self, state, cumulative_reward):
        g_n = 0
        for n in range(self.n_steps):
            # play agent turn
            # select action
            action = self.select_action(state)

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
                # opponent_action = self.opponent_action(next_state)

                self.stockfish.set_fen_position(self.env.get_epd())
                opponent_action = self.stockfish.get_best_move()

                # perform action
                next_state, reward, done = self.env.step(opponent_action)
                # Invert the reward
                if reward == 1:
                    reward = -reward

            # stats
            cumulative_reward += reward

            # cumulative discounted reward
            g_n += (self.gamma ** n) * reward
            # g_n += reward

            # update state
            state = next_state

            if done:
                break

        return first_state, first_action, state, g_n, n, done, cumulative_reward

    def learn(self):

        if len(self.replay_buffer) < self.batch_size:
            return

        batch_state, batch_action, batch_reward = self.get_transitions()
        q_values, ids, neighbor, values = self.q_network(batch_state, batch_action, True)
        # q_values = torch.zeros(self.batch_size)
        # batch_state = self.q_network(batch_state)
        # ids_list = []
        # for i, action in enumerate(batch_action):
        #     q_values[i], ids = self.memory[action] \
        #         .attend(batch_state[i].detach().view(1, -1).numpy())
        #     ids_list.append(ids)

        # q_values = self.float_tensor(q_values)

        print(batch_reward.dtype)

        calculated_loss = self.loss_fnc(q_values, batch_reward)

        self.optimizer.zero_grad()
        calculated_loss.backward()
        self.optimizer.step()
        # print(q_values.grad.data.shape)
        for nb, val, i, a in zip(neighbor, values, ids, batch_action):
            self.q_network.memory[a].learn(nb, val, i)

    # =================== Utility Functions ===================
    def select_action(self, state, evaluate=False):
        if not evaluate:
            self.steps += 1
            eps = self.eps_end + (self.eps_start - self.eps_end) * \
                  math.exp(-1. * self.steps / self.eps_decay)

        actions = self.env.get_legal_moves()

        if evaluate or random.random() > eps:
            # Greedy choice
            with torch.no_grad():
                _, q_values = self.q_network(torch.unsqueeze(state, 0), actions, False)
            idx = torch.argmax(q_values)
            return random.choice([a for i,a in enumerate(actions) if i in idx])
        else:
            # Random choice
            return random.choice(actions)

    def opponent_action(self, state):
        with torch.no_grad():
            state = self.q_network(torch.unsqueeze(state, 0))
        eps = self.opponent_end + (self.opponent_start - self.opponent_end) * math.exp(
            -1. * self.steps / self.eps_decay)
        actions = self.env.get_legal_moves()
        if random.random() > eps:
            # Greedy choice
            idx = torch.argmin(self.compute_attention(state, actions, self.memory))

            return random.choice([a for i, a in enumerate(actions) if i in idx])
        else:
            # Random choice
            return random.choice(actions)

    def get_transitions(self):
        # sample randomly from replay buffer
        transitions = self.replay_buffer.sample(self.batch_size)
        state, action, g_n = zip(*transitions)
        batch_state = Variable(torch.stack(state))
        estimated_q_vals = Variable(torch.cat(g_n))
        return batch_state, action, estimated_q_vals


if __name__ == '__main__':
    pass
