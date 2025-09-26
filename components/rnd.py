import torch
import torch.nn as nn
import torch.optim as optim

class RND(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256, output_dim=128):
        super(RND, self).__init__()

        # The target network is fixed and randomly initialized
        self.target_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # The predictor network is trained to predict the target network's output
        self.predictor_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # The target network's parameters are not updated
        for param in self.target_network.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.predictor_network.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss()

    def forward(self, obs):
        target_output = self.target_network(obs)
        predictor_output = self.predictor_network(obs)
        return predictor_output, target_output

    def train_predictor(self, obs):
        """
        Trains the predictor network to match the target network's output.
        """
        obs = torch.as_tensor(obs, dtype=torch.float32)
        predictor_output, target_output = self(obs)

        loss = self.loss_fn(predictor_output, target_output.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def compute_intrinsic_reward(self, obs):
        """
        Calculates the intrinsic reward based on the prediction error.
        """
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            predictor_output, target_output = self(obs)
            reward = self.loss_fn(predictor_output, target_output)
        return reward.item()