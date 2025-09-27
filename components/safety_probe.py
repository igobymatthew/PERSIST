import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    """Helper function to build a multi-layer perceptron."""
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

class SafetyProbe(nn.Module):
    """
    A diagnostic model that predicts the margin for each individual safety constraint
    from the agent's internal state. This provides interpretability for safety-related
    decisions.
    """
    def __init__(self, internal_dim, num_constraints, hidden_sizes=(64, 64), lr=1e-3):
        """
        Initializes the SafetyProbe.

        Args:
            internal_dim (int): The dimension of the internal state.
            num_constraints (int): The number of safety constraints to predict.
            hidden_sizes (tuple): The sizes of the hidden layers.
            lr (float): The learning rate for the optimizer.
        """
        super().__init__()
        self.net = mlp([internal_dim] + list(hidden_sizes) + [num_constraints])
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        print(f"✅ SafetyProbe initialized to predict {num_constraints} constraint margins.")

    def forward(self, internal_state):
        """
        Predicts the constraint margins for a given internal state.

        Args:
            internal_state (torch.Tensor): The internal state of the agent.

        Returns:
            torch.Tensor: The predicted margins for each constraint.
        """
        return self.net(internal_state)

    def train_probe(self, internal_states, target_margins):
        """
        Trains the probe on a batch of states and their true constraint margins.

        Args:
            internal_states (torch.Tensor): A batch of internal states.
            target_margins (torch.Tensor): The corresponding true constraint margins.
        """
        # Predict margins
        predicted_margins = self.forward(internal_states)

        # Calculate loss (MSE is suitable for regressing the margin values)
        loss = F.mse_loss(predicted_margins, target_margins)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()