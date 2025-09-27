import numpy as np
import torch

class DemonstrationBuffer:
    def __init__(self, filepath, device='cpu'):
        """
        A buffer to load and sample from expert demonstrations.

        Args:
            filepath (str): Path to the .npz file containing demonstrations.
            device (str): The torch device to move tensors to.
        """
        self.device = device
        try:
            data = np.load(filepath)
            self.observations = torch.from_numpy(data['observations']).float().to(self.device)
            self.internal_states = torch.from_numpy(data['internal_states']).float().to(self.device)
            self.size = self.observations.shape[0]
            print(f"Loaded {self.size} demonstrations from {filepath}")
        except FileNotFoundError:
            print(f"Error: Demonstration file not found at {filepath}")
            self.observations = torch.empty(0)
            self.internal_states = torch.empty(0)
            self.size = 0
        except Exception as e:
            print(f"An error occurred while loading demonstrations: {e}")
            self.observations = torch.empty(0)
            self.internal_states = torch.empty(0)
            self.size = 0

    def sample(self, batch_size):
        """
        Samples a batch of demonstrations.

        Args:
            batch_size (int): The number of samples to return.

        Returns:
            A tuple of (observations, internal_states)
        """
        if self.size == 0:
            return torch.empty(0), torch.empty(0)

        indices = np.random.randint(0, self.size, size=batch_size)
        return self.observations[indices], self.internal_states[indices]

    def __len__(self):
        return self.size