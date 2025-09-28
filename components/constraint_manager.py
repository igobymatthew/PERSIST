import torch
import torch.nn as nn
import numpy as np

class ConstraintManager:
    """
    Manages constraints using a dual ascent mechanism to automatically tune
    penalty weights (Lagrangian multipliers) for homeostatic rewards.

    Instead of a fixed `lambda_H`, this component learns a separate multiplier
    for each homeostatic constraint to maintain a target violation rate. The
    agent's total reward will be modified to include these adaptive penalties.

    The update rule for the multipliers (dual variables) is a simple
    gradient ascent step:
    lambda_{t+1} = max(0, lambda_t + lr * (current_violation - target_violation_rate))
    """
    def __init__(self, num_constraints, constraint_dim_indices, target_violation_rate=0.01, dual_lr=1e-3, device='cpu'):
        """
        Initializes the ConstraintManager.

        Args:
            num_constraints (int): The number of homeostatic constraints to manage.
            constraint_dim_indices (list[int]): A list mapping each constraint to an internal state dimension index.
            target_violation_rate (float): The desired average violation rate for each constraint.
            dual_lr (float): The learning rate for updating the dual variables (multipliers).
            device (torch.device or str): The device to store tensors on.
        """
        self.num_constraints = num_constraints
        self.target_violation_rate = target_violation_rate
        self.dual_lr = dual_lr
        self.device = device
        self.constraint_dim_indices = torch.tensor(constraint_dim_indices, device=self.device, dtype=torch.long)

        # Initialize dual variables (Lagrangian multipliers) for each constraint.
        # These are the adaptive penalty weights.
        self.lambdas = torch.zeros(self.num_constraints, requires_grad=False, device=self.device)

    def update(self, violations):
        """
        Updates the dual variables based on the observed constraint violations.

        This method should be called periodically (e.g., at the end of each episode
        or after a fixed number of steps).

        Args:
            violations (torch.Tensor): A tensor of shape (batch_size, num_constraints)
                                       where each element is 1 if the constraint was
                                       violated and 0 otherwise.
        """
        if violations.numel() == 0:
            return

        # Calculate the average violation rate for each constraint across the batch
        current_violation_rates = violations.float().mean(dim=0)

        # The error term for the dual update
        error = current_violation_rates - self.target_violation_rate

        # Perform the dual ascent step
        with torch.no_grad():
            update_amount = self.dual_lr * error
            new_lambdas = self.lambdas + update_amount
            # Project the lambdas to be non-negative
            self.lambdas = torch.clamp(new_lambdas, min=0)

    def get_penalties(self, x, mu, w):
        """
        Calculates the adaptive homeostatic penalty for the current state.

        This replaces the fixed `lambda_H * R_homeo` term. The penalty for each
        constraint is now `lambda_i * (x_i - mu_i)^2 * w_i`.

        Args:
            x (torch.Tensor): The current internal state tensor.
            mu (torch.Tensor): The setpoints for the internal state variables.
            w (torch.Tensor): The base weights for the homeostatic penalties.

        Returns:
            torch.Tensor: The total adaptive homeostatic penalty.
        """
        num_internal_dims = w.shape[0]

        # Create a tensor of effective lambdas for each internal state dimension
        # by summing the lambdas of all constraints that apply to that dimension.
        effective_lambdas = torch.zeros(num_internal_dims, device=self.device)
        effective_lambdas.scatter_add_(0, self.constraint_dim_indices, self.lambdas)

        # The penalty is the squared error weighted by the effective lambdas and base weights.
        squared_error = (x - mu)**2
        penalty = effective_lambdas * w * squared_error
        return penalty.sum(dim=-1)

    def get_lambda_history(self):
        """Returns the current values of the dual variables for logging."""
        return self.lambdas.detach().cpu().numpy().copy()