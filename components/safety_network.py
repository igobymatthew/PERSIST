import torch
import torch.nn as nn
import torch.optim as optim

class SafetyNetwork(nn.Module):
    def __init__(self, internal_dim, action_dim, hidden_dim=128):
        super(SafetyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(internal_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # To keep the output in a reasonable range, assuming actions are scaled
        )
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, internal_state, unsafe_action):
        input_tensor = torch.cat([internal_state, unsafe_action], dim=-1)
        return self.net(input_tensor)

    def train_network(self, unsafe_actions, safe_actions, internal_states):
        """
        Trains the network to predict the safe action from an unsafe one.
        """
        self.optimizer.zero_grad()
        predicted_safe_actions = self.forward(internal_states, unsafe_actions)
        loss = nn.MSELoss()(predicted_safe_actions, safe_actions)
        loss.backward()
        self.optimizer.step()
        return loss.item()