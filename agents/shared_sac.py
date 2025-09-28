import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

# Re-using the MLP builder from the original SAC implementation
def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SharedCritic(nn.Module):
    """A critic that takes observation, action, and a role embedding."""
    def __init__(self, obs_dim, act_dim, embedding_dim, hidden_sizes, activation):
        super().__init__()
        # The input dimension is the sum of observation, action, and embedding dimensions
        input_dim = obs_dim + act_dim + embedding_dim
        self.net = mlp([input_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act, embedding):
        x = torch.cat([obs, act, embedding], dim=-1)
        return self.net(x)

class SharedSquashedGaussianActor(nn.Module):
    """An actor that takes observation and a role embedding."""
    def __init__(self, obs_dim, act_dim, embedding_dim, hidden_sizes, activation):
        super().__init__()
        # The input dimension is the sum of observation and embedding dimensions
        input_dim = obs_dim + embedding_dim
        self.net = mlp([input_dim] + list(hidden_sizes) + [2 * act_dim], activation, nn.Identity)

    def forward(self, obs, embedding, deterministic=False, with_logprob=True):
        x = torch.cat([obs, embedding], dim=-1)
        net_out = self.net(x)
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

class SharedSAC(nn.Module):
    """
    A parameter-sharing SAC agent for homogeneous multi-agent settings.
    It uses role embeddings to allow a single policy to be used by multiple agents.
    """
    def __init__(self, obs_dim, act_dim, num_agents, role_embedding_dim, hidden_sizes=(256, 256), activation=nn.ReLU, config=None):
        super().__init__()
        self.config = config if config else {}
        self.num_agents = num_agents
        self.agent_ids = list(range(num_agents)) # Simple integer IDs for embedding lookup

        # Create a learnable embedding for each agent
        self.role_embedding = nn.Embedding(num_agents, role_embedding_dim)

        # Create the shared actor and critics
        self.actor = SharedSquashedGaussianActor(obs_dim, act_dim, role_embedding_dim, hidden_sizes, activation)
        self.critic1 = SharedCritic(obs_dim, act_dim, role_embedding_dim, hidden_sizes, activation)
        self.critic2 = SharedCritic(obs_dim, act_dim, role_embedding_dim, hidden_sizes, activation)

        # Create target networks
        self.critic1_target = SharedCritic(obs_dim, act_dim, role_embedding_dim, hidden_sizes, activation)
        self.critic2_target = SharedCritic(obs_dim, act_dim, role_embedding_dim, hidden_sizes, activation)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Alpha (entropy regularization)
        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.target_entropy = -torch.prod(torch.Tensor((act_dim,))).item()

        # Optimizers
        training_cfg = self.config.get('training', {})
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=training_cfg.get('actor_lr', 1e-3))
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=training_cfg.get('critic_lr', 1e-3))
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=training_cfg.get('actor_lr', 1e-3)) # use actor lr for alpha

    def get_action(self, obs, agent_id, deterministic=False):
        """Get an action for a single agent."""
        with torch.no_grad():
            device = next(self.actor.parameters()).device
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            agent_id_tensor = torch.as_tensor([agent_id], dtype=torch.long, device=device)
            embedding = self.role_embedding(agent_id_tensor)

            action, _ = self.actor(obs_tensor, embedding, deterministic, False)
            return action.squeeze(0).cpu().numpy()

    def update(self, data):
        """
        Update the actor and critic networks for all agents using a shared batch.
        """
        obs, act, rew, next_obs, done, agent_ids = \
            data['obs'], data['act'], data['rew'], data['next_obs'], data['done'], data['agent_ids']

        device = next(self.actor.parameters()).device
        agent_ids_tensor = torch.as_tensor(agent_ids, dtype=torch.long, device=device)
        embeddings = self.role_embedding(agent_ids_tensor)

        gamma = self.config.get('training', {}).get('gamma', 0.99)
        polyak = self.config.get('training', {}).get('tau', 0.995)
        alpha = torch.exp(self.log_alpha.detach())

        # --- Critic Update ---
        with torch.no_grad():
            next_act, next_logp = self.actor(next_obs, embeddings)

            q1_target = self.critic1_target(next_obs, next_act, embeddings)
            q2_target = self.critic2_target(next_obs, next_act, embeddings)
            q_target = torch.min(q1_target, q2_target).squeeze(-1)

            backup = rew + gamma * (1 - done) * (q_target - alpha * next_logp)

        q1 = self.critic1(obs, act, embeddings).squeeze(-1)
        q2 = self.critic2(obs, act, embeddings).squeeze(-1)

        critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor and Alpha Update ---
        for p in self.critic1.parameters(): p.requires_grad = False
        for p in self.critic2.parameters(): p.requires_grad = False

        pi, logp_pi = self.actor(obs, embeddings)
        q1_pi = self.critic1(obs, pi, embeddings)
        q2_pi = self.critic2(obs, pi, embeddings)
        q_pi = torch.min(q1_pi, q2_pi).squeeze(-1)

        actor_loss = (alpha * logp_pi - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = (-self.log_alpha * (logp_pi.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for p in self.critic1.parameters(): p.requires_grad = True
        for p in self.critic2.parameters(): p.requires_grad = True

        # --- Target Network Update ---
        with torch.no_grad():
            for p, p_target in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                p_target.data.mul_(polyak)
                p_target.data.add_((1 - polyak) * p.data)
            for p, p_target in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                p_target.data.mul_(polyak)
                p_target.data.add_((1 - polyak) * p.data)

        return -logp_pi.detach().mean().item()