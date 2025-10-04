# persist/evolution/evaluators/rl_metaeval.py
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, input_dim, output_dim, net_width, net_depth):
        super(Policy, self).__init__()
        layers = [nn.Linear(input_dim, net_width), nn.ReLU()]
        for _ in range(net_depth - 1):
            layers.extend([nn.Linear(net_width, net_width), nn.ReLU()])
        layers.append(nn.Linear(net_width, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


def train_and_measure(cfg, max_episodes=100):
    """
    Trains a simple REINFORCE agent on CartPole and returns metrics.
    """
    start_time = time.time()

    env = gym.make("CartPole-v1")
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    policy = Policy(input_dim, output_dim, cfg["net_width"], cfg["net_depth"])
    optimizer = optim.Adam(policy.parameters(), lr=cfg["lr"])

    all_rewards = []

    for episode in range(max_episodes):
        saved_log_probs = []
        rewards = []
        state, _ = env.reset()

        for t in range(1000): # Max steps per episode
            action_probs = policy(torch.tensor(state, dtype=torch.float32))
            m = Categorical(action_probs)
            action = m.sample()
            saved_log_probs.append(m.log_prob(action))
            state, reward, done, _, _ = env.step(action.item())
            rewards.append(reward)
            if done:
                break

        all_rewards.append(sum(rewards))

        # REINFORCE update
        R = 0
        policy_loss = []
        returns = []
        for r in reversed(rewards):
            R = r + cfg.get("gamma", 0.99) * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        for log_prob, R in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        optimizer.zero_grad()
        if policy_loss:
            policy_loss = torch.stack(policy_loss).sum()
            policy_loss.backward()
            optimizer.step()

    latency = time.time() - start_time
    avg_return = np.mean(all_rewards)

    # CartPole doesn't have constraints, so violations are 0
    violations = 0.0

    return avg_return, violations, latency


def eval_rl_individual(ind) -> dict:
    """
    Evaluates an individual by training an RL agent with its genes as config.
    """
    cfg = ind["genes"]  # net width/depth, lr, etc.
    ep_return, violations, latency = train_and_measure(cfg)
    return {"neg_return": -ep_return, "viol": violations, "latency": latency}