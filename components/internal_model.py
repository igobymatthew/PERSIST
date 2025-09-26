import torch
import torch.nn as nn
import torch.optim as optim

class InternalModel(nn.Module):
    def __init__(self, internal_dim, act_dim, hidden_dim=256):
        super(InternalModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(internal_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, internal_dim)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

    def forward(self, x, act):
        combined = torch.cat([x, act], dim=-1)
        return self.net(combined)

    def train_model(self, x, act, next_x):
        x = torch.as_tensor(x, dtype=torch.float32)
        act = torch.as_tensor(act, dtype=torch.float32)
        next_x = torch.as_tensor(next_x, dtype=torch.float32)

        predicted_next_x = self(x, act)
        loss = self.loss_fn(predicted_next_x, next_x)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict_next(self, x, act):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
            act = torch.as_tensor(act, dtype=torch.float32).unsqueeze(0)
            predicted_next_x = self(x, act)
        return predicted_next_x.squeeze(0).numpy()