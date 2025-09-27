import torch
import torch.nn as nn
import torch.optim as optim

class ViabilityApproximator(nn.Module):
    def __init__(self, internal_dim, hidden_dim=256, lr=1e-3):
        super(ViabilityApproximator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(internal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output a value between 0 (unsafe) and 1 (safe)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.BCELoss() # Binary Cross-Entropy for classification

    def forward(self, x):
        # Ensure input is a tensor and on the correct device
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)

        device = next(self.parameters()).device
        if x.device != device:
            x = x.to(device)
        return self.net(x)

    def train_model(self, x, labels):
        """
        Performs one training step.
        Args:
            x (Tensor): Input states.
            labels (Tensor): Target labels (0 for unsafe, 1 for safe).
        """
        device = next(self.parameters()).device
        # Ensure labels are a tensor and on the correct device
        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels, dtype=torch.float32)
        if labels.device != device:
            labels = labels.to(device)

        # Ensure labels have the correct shape (batch_size, 1)
        if len(labels.shape) == 1:
            labels = labels.unsqueeze(1)

        margin = self(x)
        loss = self.loss_fn(margin, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_on_demonstrations(self, demo_buffer, batch_size):
        """
        Trains the model for one step using a batch of safe demonstrations.
        All states from the demonstration buffer are considered safe (label=1).
        """
        if len(demo_buffer) == 0:
            return 0.0

        _, internal_states = demo_buffer.sample(batch_size)

        # All demonstration states are considered safe, so the label is 1.
        labels = torch.ones(internal_states.shape[0])

        return self.train_model(internal_states, labels)

    def get_margin(self, x):
        """
        Gets the safety margin for a given state x.
        Handles both single samples and batches.
        """
        was_unbatched = False
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            was_unbatched = True

        with torch.no_grad():
            margin = self(x)

        return margin.squeeze(0) if was_unbatched else margin