import numpy as np

class Homeostat:
    def __init__(self, mu, w):
        self.mu = np.array(mu)
        self.w = np.array(w)

    def reward(self, x):
        return -np.sum((x - self.mu)**2 * self.w, axis=-1)