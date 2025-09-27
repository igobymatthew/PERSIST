import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, output_activation):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, output_activation)

    def forward(self, obs):
        return self.net(obs)

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.net = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianActor(Actor):
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

class SAC(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()

        self.actor = SquashedGaussianActor(obs_dim, act_dim, hidden_sizes, activation)
        self.critic1 = Critic(obs_dim, act_dim, hidden_sizes, activation)
        self.critic2 = Critic(obs_dim, act_dim, hidden_sizes, activation)

        self.critic1_target = Critic(obs_dim, act_dim, hidden_sizes, activation)
        self.critic2_target = Critic(obs_dim, act_dim, hidden_sizes, activation)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.target_entropy = -torch.prod(torch.Tensor((act_dim,))).item()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=1e-3)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-3)

    def step(self, obs, deterministic=False):
        return self.get_action(obs, deterministic)

    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            device = next(self.actor.parameters()).device
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            action, _ = self.actor(obs_tensor, deterministic, False)
            return action.cpu().numpy()

    def learn(self, data, gamma=0.99, polyak=0.995, ewc_penalty=0.0, adversary=None):
        obs, act, rew, next_obs, done = data['obs'], data['action'], data['reward'], data['next_obs'], data['done']

        alpha = torch.exp(self.log_alpha)

        # --- Critic Update ---
        with torch.no_grad():
            next_act, next_logp = self.actor(next_obs)
            q1_target = self.critic1_target(next_obs, next_act)
            q2_target = self.critic2_target(next_obs, next_act)
            q_target = torch.min(q1_target, q2_target).squeeze(-1)
            backup = rew + gamma * (1 - done) * (q_target - alpha * next_logp)

        # Adversarial training step for the critic
        obs_for_critic = obs
        if adversary:
            # Generate adversarial observation to attack the critic
            obs_for_critic = adversary.perturb(self.critic1, obs, act, backup)

        q1 = self.critic1(obs_for_critic, act).squeeze(-1)
        q2 = self.critic2(obs_for_critic, act).squeeze(-1)

        critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor and Alpha Update ---
        for p in self.critic1.parameters():
            p.requires_grad = False
        for p in self.critic2.parameters():
            p.requires_grad = False

        # Actor learns from the original, non-perturbed state
        pi, logp_pi = self.actor(obs)
        q1_pi = self.critic1(obs, pi)
        q2_pi = self.critic2(obs, pi)
        q_pi = torch.min(q1_pi, q2_pi).squeeze(-1)

        # Add the EWC penalty to the actor loss
        actor_loss = (alpha.detach() * logp_pi - q_pi).mean() + ewc_penalty

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = (-self.log_alpha * (logp_pi.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for p in self.critic1.parameters():
            p.requires_grad = True
        for p in self.critic2.parameters():
            p.requires_grad = True

        # --- Target Network Update ---
        with torch.no_grad():
            for p, p_target in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                p_target.data.mul_(polyak)
                p_target.data.add_((1 - polyak) * p.data)
            for p, p_target in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                p_target.data.mul_(polyak)
                p_target.data.add_((1 - polyak) * p.data)

        # Return policy entropy for logging
        return -logp_pi.detach().mean().item()