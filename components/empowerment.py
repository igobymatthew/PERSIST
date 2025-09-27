import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    """
    A discriminator for empowerment, mapping (state, action_sequence, future_state) to a scalar.
    """
    def __init__(self, state_dim: int, action_dim: int, k: int, hidden_dim: int = 256):
        """
        Args:
            state_dim: The dimension of the latent state.
            action_dim: The dimension of the action space.
            k: The number of steps in the action sequence.
            hidden_dim: The hidden dimension size.
        """
        super().__init__()
        # The input is the concatenation of the current latent state, the future latent state,
        # and the flattened action sequence that connects them.
        input_dim = state_dim + state_dim + (action_dim * k)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, action_sequence: torch.Tensor, future_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the discriminator.

        Args:
            state: The starting latent state (batch_size, state_dim).
            action_sequence: The sequence of actions (batch_size, k, action_dim).
            future_state: The resulting latent state (batch_size, state_dim).

        Returns:
            A scalar logit for each sample in the batch (batch_size, 1).
        """
        # Flatten the action sequence
        action_sequence_flat = action_sequence.view(action_sequence.size(0), -1)

        # Concatenate all inputs
        x = torch.cat([state, action_sequence_flat, future_state], dim=-1)
        return self.net(x)


class Empowerment(nn.Module):
    """
    Calculates empowerment-based intrinsic reward using an InfoNCE-style contrastive loss.
    """
    def __init__(self, state_dim: int, action_dim: int, k: int, hidden_dim: int = 256, lr: float = 1e-4, device: str = 'cpu'):
        """
        Args:
            state_dim: The dimension of the latent state.
            action_dim: The dimension of the action space.
            k: The number of steps in the action sequence for empowerment calculation.
            hidden_dim: The hidden dimension size for the discriminator.
            lr: The learning rate for the discriminator's optimizer.
            device: The torch device to use.
        """
        super().__init__()
        self.discriminator = Discriminator(state_dim, action_dim, k, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        self.k = k
        self.device = device

    def compute_reward(self, state: torch.Tensor, action_sequence: torch.Tensor, future_state: torch.Tensor) -> torch.Tensor:
        """
        Computes the intrinsic reward for a given transition.
        The reward is the log-probability of the discriminator, which is approximated by its raw logit output.
        """
        with torch.no_grad():
            score = self.discriminator(state, action_sequence, future_state)
            return score

    def update(self, state: torch.Tensor, action_sequence: torch.Tensor, future_state: torch.Tensor) -> float:
        """
        Updates the discriminator using a contrastive (InfoNCE) loss.
        The goal is to distinguish between true future states and negative samples.

        Args:
            state: The starting latent state (batch_size, state_dim).
            action_sequence: The sequence of actions (batch_size, k, action_dim).
            future_state: The true resulting latent state (batch_size, state_dim).

        Returns:
            The value of the loss for logging.
        """
        batch_size = future_state.size(0)

        # Positive samples: the true (state, action_sequence, future_state) tuples
        positive_scores = self.discriminator(state, action_sequence, future_state)

        # Negative samples: pair states with future states from other trajectories in the batch
        # A simple and effective way to generate negatives is to shuffle the future states.
        shuffled_indices = torch.randperm(batch_size, device=self.device)
        negative_future_states = future_state[shuffled_indices]

        negative_scores = self.discriminator(state, action_sequence, negative_future_states)

        # Create labels for binary cross-entropy loss
        positive_labels = torch.ones_like(positive_scores)
        negative_labels = torch.zeros_like(negative_scores)

        # Concatenate positive and negative samples
        scores = torch.cat([positive_scores, negative_scores], dim=0)
        labels = torch.cat([positive_labels, negative_labels], dim=0)

        # Calculate binary cross-entropy loss with logits
        loss = F.binary_cross_entropy_with_logits(scores, labels)

        # Update discriminator
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()