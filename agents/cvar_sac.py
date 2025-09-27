import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    """Creates a multi-layer perceptron."""
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class Actor(nn.Module):
    """Base class for the policy network."""
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, output_activation):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, output_activation)

    def forward(self, obs):
        return self.net(obs)

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianActor(Actor):
    """
    A Gaussian policy network with a squashing function (tanh) to bound the output.
    This is the standard actor for SAC.
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__(obs_dim, 2 * act_dim, hidden_sizes, activation, nn.Identity)

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu, log_std = torch.chunk(net_out, 2, dim=-1)

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)

        return pi_action, logp_pi

class DistributionalCritic(nn.Module):
    """
    A critic network that outputs a distribution of Q-values (quantiles)
    for a given state-action pair. This is the core of the risk-sensitive approach.
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, n_quantiles, activation=nn.ReLU):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.net = mlp([obs_dim + act_dim] + list(hidden_sizes) + [n_quantiles], activation)

    def forward(self, obs, act):
        """
        Returns the predicted quantiles for the given state-action pair.
        Shape: (batch_size, n_quantiles)
        """
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)

class CVAR_SAC(nn.Module):
    """
    A risk-sensitive version of SAC that optimizes the CVaR of the return distribution.
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256), activation=nn.ReLU,
                 n_quantiles=32, tau=0.1):
        super().__init__()

        self.actor = SquashedGaussianActor(obs_dim, act_dim, hidden_sizes, activation)
        self.critic1 = DistributionalCritic(obs_dim, act_dim, hidden_sizes, n_quantiles, activation)
        self.critic2 = DistributionalCritic(obs_dim, act_dim, hidden_sizes, n_quantiles, activation)

        self.critic1_target = DistributionalCritic(obs_dim, act_dim, hidden_sizes, n_quantiles, activation)
        self.critic2_target = DistributionalCritic(obs_dim, act_dim, hidden_sizes, n_quantiles, activation)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.target_entropy = -torch.prod(torch.Tensor((act_dim,))).item()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=1e-3)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-3)

        self.n_quantiles = n_quantiles
        self.tau = tau  # Risk-aversion parameter for CVaR

        # Define the quantiles (cumulative probabilities)
        # These are the midpoints of the n_quantiles intervals
        self.cumulative_probs = (torch.arange(n_quantiles, dtype=torch.float32) + 0.5) / n_quantiles

    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            action, _ = self.actor(obs_tensor, deterministic, False)
            return action.numpy()

    def quantile_regression_loss(self, current_quantiles, target_quantiles):
        """
        Calculates the quantile regression loss.
        """
        # Ensure target_quantiles are detached from the graph
        target_quantiles = target_quantiles.detach()
        # Reshape for broadcasting
        # current_quantiles: (batch_size, n_quantiles)
        # target_quantiles: (batch_size, n_quantiles)
        # pairwise_delta: (batch_size, n_quantiles, n_quantiles)
        pairwise_delta = target_quantiles.unsqueeze(-2) - current_quantiles.unsqueeze(-1)
        abs_pairwise_delta = torch.abs(pairwise_delta)
        # Huber loss
        huber_loss = torch.where(abs_pairwise_delta > 1.0, abs_pairwise_delta - 0.5, pairwise_delta**2 * 0.5)

        # Quantile regression loss
        # The (cumulative_probs.view(1, -1) - (pairwise_delta < 0).float()) term is the key part
        # It weights errors based on the quantile
        loss = torch.abs(self.cumulative_probs.view(1, -1) - (pairwise_delta < 0).float()) * huber_loss
        return loss.mean()

    def update(self, data, gamma=0.99, polyak=0.995):
        obs = torch.as_tensor(data['obs'], dtype=torch.float32)
        act = torch.as_tensor(data['action'], dtype=torch.float32)
        rew = torch.as_tensor(data['reward'], dtype=torch.float32)
        next_obs = torch.as_tensor(data['next_obs'], dtype=torch.float32)
        done = torch.as_tensor(data['done'], dtype=torch.float32)

        alpha = torch.exp(self.log_alpha).detach()

        # Critic update
        with torch.no_grad():
            next_act, next_logp = self.actor(next_obs)
            # Get target quantile distributions
            q1_target = self.critic1_target(next_obs, next_act)
            q2_target = self.critic2_target(next_obs, next_act)
            # Take the minimum of the two distributions element-wise
            q_target = torch.min(q1_target, q2_target)

            # Apply entropy correction to the target distribution
            # Each quantile is shifted by the same amount
            q_target_entropy_corrected = q_target - alpha * next_logp.unsqueeze(-1)

            # Compute the Bellman backup for each quantile
            backup = rew.unsqueeze(-1) + gamma * (1 - done).unsqueeze(-1) * q_target_entropy_corrected

        # Get current quantile distributions
        q1 = self.critic1(obs, act)
        q2 = self.critic2(obs, act)

        # Calculate quantile regression loss for both critics
        critic1_loss = self.quantile_regression_loss(q1, backup)
        critic2_loss = self.quantile_regression_loss(q2, backup)
        critic_loss = critic1_loss + critic2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor and alpha update
        for p in self.critic1.parameters():
            p.requires_grad = False
        for p in self.critic2.parameters():
            p.requires_grad = False

        pi, logp_pi = self.actor(obs)
        # Get the quantile distribution for the new action
        q1_pi = self.critic1(obs, pi)
        q2_pi = self.critic2(obs, pi)
        q_pi_dist = torch.min(q1_pi, q2_pi)

        # Calculate CVaR from the distribution
        # This is the mean of the quantiles up to the tau-th quantile
        num_quantiles_for_cvar = int(self.n_quantiles * self.tau)
        cvar = q_pi_dist[:, :num_quantiles_for_cvar].mean(dim=1)

        # Actor loss is the negative of the CVaR (we want to maximize it) plus entropy
        actor_loss = (alpha * logp_pi - cvar).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha (temperature) update
        alpha_loss = (-self.log_alpha * (logp_pi.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for p in self.critic1.parameters():
            p.requires_grad = True
        for p in self.critic2.parameters():
            p.requires_grad = True

        # Target network update
        with torch.no_grad():
            for p, p_target in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                p_target.data.mul_(polyak)
                p_target.data.add_((1 - polyak) * p.data)
            for p, p_target in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                p_target.data.mul_(polyak)
                p_target.data.add_((1 - polyak) * p.data)

        # Return policy entropy for logging
        return -logp_pi.detach().mean().item()