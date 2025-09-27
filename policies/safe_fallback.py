import torch
import torch.nn as nn

class SafeFallbackPolicy(nn.Module):
    """
    A simple, rule-based policy that provides a safe, default action.

    This policy is intended to be used as a fallback when the main agent
    encounters an Out-of-Distribution (OOD) state or another critical failure
    condition where its own output cannot be trusted.

    The "safe" action is defined as a zero action, which in many environments
    corresponds to a no-op, braking, or maintaining the current state. This
    is a generic but often effective strategy for immediate risk mitigation.
    """
    def __init__(self, action_dim, device='cpu'):
        """
        Initializes the SafeFallbackPolicy.

        Args:
            action_dim (int): The dimensionality of the action space.
            device (torch.device or str): The device to create the action tensor on.
        """
        super().__init__()
        self.action_dim = action_dim
        self.device = device

    def get_action(self, batch_size=1):
        """
        Returns a batch of safe, zero-actions.

        Args:
            batch_size (int): The number of actions to return in the batch.

        Returns:
            torch.Tensor: A tensor of zero-actions. Shape (batch_size, action_dim).
        """
        return torch.zeros((batch_size, self.action_dim), device=self.device)

    def forward(self, batch_size=1):
        """Convenience forward method."""
        return self.get_action(batch_size)