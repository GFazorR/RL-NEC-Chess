import numpy as np
import torch
from sklearn.neighbors import KDTree


class DifferentiableNeuralDictionary:
    def __init__(self, size, key_size, k):
        self.embeddings = np.zeros((size, key_size))
        self.q_values = np.zeros(size)
        self.lru = np.zeros(size)
        self.tm = .1
        self.k = k
        self.tree = None
        self.current_size = 0
        self.max_size = size

    def lookup(self, key):
        key = key.clone().detach().numpy()
        idx = self.tree.query(key, k=self.k, return_distance=False)
        self.lru += self.tm
        return torch.from_numpy(self.embeddings[idx].reshape((10, 64))), torch.from_numpy(self.q_values[idx])

    def write(self, key, value):
        key = key.clone().detach().numpy()
        value = value.clone().detach().numpy()
        index = (self.embeddings == key).all(axis=1).nonzero()
        if len(index[0])> 0:
            self.q_values[index[0][0]] = value
        elif self.current_size < self.max_size:
            self.embeddings[self.current_size] = key
            self.q_values[self.current_size] = value
            self.current_size += 1
            self.rebuild_tree()
        else:
            index = np.argmin(self.lru)
            self.embeddings[index] = key
            self.q_values[index] = value

            self.rebuild_tree()

    def update(self, gamma, g_n, state, kernel):
        w = self.calculate_weights(state, kernel)
        self.q_values[:self.current_size] = self.q_values[:self.current_size] \
                                           + gamma * w * (g_n - self.q_values[:self.current_size])
    def calculate_weights(self, state, kernel):
        # compute weights
        torch_embeddings = torch.from_numpy(self.embeddings[:self.current_size])
        kernel_sum = kernel(torch_embeddings, state)
        w = kernel_sum / torch.sum(kernel_sum)
        return w.detach().numpy()

    def rebuild_tree(self):
        self.tree = KDTree(self.embeddings[:self.current_size])

    def is_queryable(self):
        return self.current_size >= self.k


if __name__ == '__main__':
    pass