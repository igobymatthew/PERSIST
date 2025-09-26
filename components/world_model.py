import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class WorldModel(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super(WorldModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)

    def train_model(self, obs, act, next_obs):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        act = torch.as_tensor(act, dtype=torch.float32)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32)

        predicted_next_obs = self(obs, act)
        loss = self.loss_fn(predicted_next_obs, next_obs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def compute_surprise_reward(self, obs, act, next_obs):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            act = torch.as_tensor(act, dtype=torch.float32).unsqueeze(0)
            next_obs = torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0)

            predicted_next_obs = self(obs, act)
            surprise = self.loss_fn(predicted_next_obs, next_obs)
        return surprise.item()