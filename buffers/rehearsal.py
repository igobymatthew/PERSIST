import torch
import random
import numpy as np

class RehearsalBuffer:
    """
    A simple buffer to store states for continual learning.
    It uses reservoir sampling to maintain a fixed-size, unbiased sample
    of the states encountered so far.
    """
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device # device is not used here but good to keep for consistency
        self.buffer = []
        self.n_seen = 0

    def add(self, state):
        """
        Adds a state to the buffer using reservoir sampling.

        Args:
            state (np.ndarray): The state observation for the agent.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(state)
        else:
            idx = random.randint(0, self.n_seen)
            if idx < self.capacity:
                self.buffer[idx] = state
        self.n_seen += 1

    def sample(self, batch_size):
        """
        Samples a batch of states from the buffer.

        Args:
            batch_size (int): The number of states to sample.

        Returns:
            A list of states (numpy arrays).
        """
        if not self.buffer:
            return []
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)