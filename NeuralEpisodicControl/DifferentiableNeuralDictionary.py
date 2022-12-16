import numpy as np
import sklearn.neighbors
from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
import torch
from torch import nn


class DifferentiableNeuralDictionary(nn.Module):
    def __init__(self, size, key_size, k=10):
        super().__init__()
        self.embeddings = np.zeros((size, key_size))
        self.key_size = key_size
        self.q_values = np.zeros(size)
        self.lru = np.zeros(size)
        self.tm = .1
        self.k = k
        self.tree = None
        self.current_size = 0
        self.max_size = size
        self.kernel = cosine_similarity
        self.key_square_avg = torch.zeros(self.max_size, self.key_size)
        self.q_values_avg = torch.zeros(self.max_size, self.key_size)

    def lookup(self, key):
        if self.current_size < self.k:
            k = self.current_size
        else:
            k = self.k
        d, idx = self.tree.query(key, k=k, return_distance=True)
        self.lru[idx] += self.tm

        # distances = self.kernel(self.embeddings[idx[0]], key)
        return 1 - d, idx, self.q_values[idx]

    def write(self, key, value):
        if self.current_size == 0:
            self.embeddings[self.current_size] = key
            self.q_values[self.current_size] = value
            self.current_size += 1
            self.rebuild_tree()
            return
        d, idx = self.tree.query(key, k=1, return_distance=True)
        if d[0][0] == 0:
            self.q_values[idx[0][0]] = value
        elif self.current_size < self.max_size:
            self.embeddings[self.current_size] = key
            self.q_values[self.current_size] = value
            self.current_size += 1
            self.rebuild_tree()
        else:
            index = np.argmax(self.lru)
            self.embeddings[index] = key
            self.q_values[index] = value
            self.lru[index] = 0

            self.rebuild_tree()

    def update(self, gamma, g_n, state):
        distances, ids, _ = self.lookup(state)
        # distances = self.kernel(self.embeddings[:self.current_size], state)
        w = self.calculate_weights(distances, state)
        self.q_values[ids[0]] = \
            self.q_values[ids[0]] \
            + gamma * w * (g_n - self.q_values[ids[0]])

    def calculate_weights(self, distances, state):
        # compute weights
        all_distances, _ = self.tree.query(state, k=self.current_size, return_distance=True)
        w = distances / np.sum(all_distances)
        return w

    def forward(self, state, learning):
        distances, ids, values = self.lookup(state.detach().numpy())
        neighbours = torch.tensor(self.embeddings[ids], requires_grad=True)
        w = torch.tensor(self.calculate_weights(distances, state.detach().numpy()))
        values = torch.tensor(values, requires_grad=True)
        self.lru += 1
        self.lru[ids[0]] = 0
        out = torch.sum(w * values)
        if learning:
            return out, ids, neighbours, values
        else:
            return out
    def learn(self, keys, values, ids):
        # update keys
        if keys.grad is not None:
            keys_grad = keys.grad.data
            key_update = self.key_square_avg[ids]
            key_update += keys_grad.mul(keys_grad)
            avg = key_update.sqrt()
            keys.data.addcdiv(1e-3, keys_grad, avg)
            self.embeddings[ids] = keys.detach().numpy()
            self.key_square_avg[ids] = key_update
        if values.grad is not None:
            values_grad = values.grad.data
            values_update = torch.zeros(self.max_size, 1)
            values_update += values_grad.mul(values_grad)
            avg = values_update.sqrt()
            values.data.addcdiv(1e-3, values_grad, avg)
            self.q_values[ids[0]] = values.detach().numpy()
            self.q_values_avg[ids[0]] = values_update

    def get_closest_distance(self, state):
        # d = np.max(self.kernel(self.embeddings[:self.current_size], state))
        d, _ = self.tree.query(state, k=1, return_distance=True)
        # print(d)
        return d[0][0]

    def rebuild_tree(self):
        self.tree = KDTree(self.embeddings[:self.current_size], metric='euclidean')

    def is_queryable(self):
        return self.current_size > 0


if __name__ == '__main__':
    # a = np.array([[1, 2, 3], [7, 5, 4], [7, 1, 9]])
    # b = np.array([[1, 2, 3]])
    print(sklearn.neighbors.VALID_METRICS['kd_tree'])
    pass
