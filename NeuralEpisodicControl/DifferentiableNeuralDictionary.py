import numpy as np
from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import rbf_kernel


class DifferentiableNeuralDictionary:
    def __init__(self, size, key_size, k=50):
        self.embeddings = np.zeros((size, key_size))
        self.key_size = key_size
        self.q_values = np.zeros(size)
        self.lru = np.zeros(size)
        self.tm = .1
        self.k = k
        self.tree = None
        self.current_size = 0
        self.max_size = size
        self.kernel = rbf_kernel

    def lookup(self, key):
        if self.current_size < self.k:
            k = self.current_size
        else:
            k = self.k
        idx = self.tree.query(key, k=k, return_distance=False)
        self.lru[idx[0]] += self.tm
        distances = self.kernel(self.embeddings[idx[0]], key)

        return distances, idx[0], self.q_values[idx]

    def write(self, key, value):
        index = (self.embeddings == key).all(axis=1).nonzero()
        if len(index[0]) > 0:
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
            self.lru[index] = 0

            self.rebuild_tree()

    def update(self, gamma, g_n, state):

        distances = self.kernel(self.embeddings[:self.current_size], state)
        w = self.calculate_weights(distances)
        self.q_values[:self.current_size] = \
            self.q_values[:self.current_size] \
            + gamma * w * (g_n - self.q_values[:self.current_size])

    @staticmethod
    def calculate_weights(distances):
        # compute weights
        w = distances / np.sum(distances)
        return w.T

    def attend(self, state):
        distances, _, values = self.lookup(state)
        w = self.calculate_weights(distances)
        return np.sum(w * values)

    def get_closest_distance(self, state):
        d = np.max(self.kernel(self.embeddings, state))
        return d

    def rebuild_tree(self):
        self.tree = KDTree(self.embeddings[:self.current_size])

    def is_queryable(self):
        return self.current_size > 0


if __name__ == '__main__':
    pass
