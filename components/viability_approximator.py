import torch
import torch.nn as nn
import torch.optim as optim

class ViabilityApproximator(nn.Module):
    def __init__(self, internal_dim, hidden_dim=256):
        super(ViabilityApproximator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(internal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output a value between 0 (unsafe) and 1 (safe)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.loss_fn = nn.BCELoss() # Binary Cross-Entropy for classification

    def forward(self, x):
        return self.net(x)

    def train_model(self, x, labels):
        x = torch.as_tensor(x, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.float32).unsqueeze(1)

        margin = self(x)
        loss = self.loss_fn(margin, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_margin(self, x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
            margin = self(x)
        return margin.squeeze(0)