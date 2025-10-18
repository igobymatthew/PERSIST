from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kl(
    mean_q: torch.Tensor, std_q: torch.Tensor, mean_p: torch.Tensor, std_p: torch.Tensor
) -> torch.Tensor:
    """Compute the KL divergence KL(q||p) for diagonal Gaussians."""
    var_q = std_q.pow(2)
    var_p = std_p.pow(2)
    # Avoid numerical issues by working in log-space when possible.
    log_std_ratio = torch.log(std_q.clamp_min(1e-8)) - torch.log(std_p.clamp_min(1e-8))
    kl = (
        log_std_ratio
        + (var_q + (mean_q - mean_p).pow(2)) / (2.0 * var_p.clamp_min(1e-8))
        - 0.5
    )
    return kl.sum(dim=-1)


@dataclass
class RSSMState:
    mean: torch.Tensor
    std: torch.Tensor
    stoch: torch.Tensor
    deter: torch.Tensor

    def detach(self) -> "RSSMState":
        return RSSMState(
            mean=self.mean.detach(),
            std=self.std.detach(),
            stoch=self.stoch.detach(),
            deter=self.deter.detach(),
        )


class RSSM(nn.Module):
    """Recurrent state-space model with stochastic and deterministic parts."""

    def __init__(
        self,
        action_dim: int,
        embed_dim: int,
        stoch_dim: int = 32,
        deter_dim: int = 200,
        hidden_dim: int = 200,
        min_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim
        self.min_std = min_std

        self.input_layer = nn.Linear(stoch_dim + action_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, deter_dim)

        self.prior_layer = nn.Linear(deter_dim, 2 * stoch_dim)
        self.posterior_layer = nn.Linear(deter_dim + hidden_dim, 2 * stoch_dim)
        self.embed_layer = nn.Linear(embed_dim, hidden_dim)

    def init_state(self, batch_size: int, device: torch.device) -> RSSMState:
        zeros = torch.zeros(batch_size, self.stoch_dim, device=device)
        deter = torch.zeros(batch_size, self.deter_dim, device=device)
        return RSSMState(mean=zeros, std=zeros + 1.0, stoch=zeros, deter=deter)

    def get_feat(self, state: RSSMState) -> torch.Tensor:
        return torch.cat([state.stoch, state.deter], dim=-1)

    def img_step(self, prev_state: RSSMState, prev_action: torch.Tensor) -> RSSMState:
        x = torch.cat([prev_state.stoch, prev_action], dim=-1)
        hidden = torch.tanh(self.input_layer(x))
        deter = self.gru(hidden, prev_state.deter)
        stats = self.prior_layer(deter)
        mean, std = stats.chunk(2, dim=-1)
        std = F.softplus(std) + self.min_std
        stoch = mean + std * torch.randn_like(mean)
        return RSSMState(mean=mean, std=std, stoch=stoch, deter=deter)

    def obs_step(
        self,
        prev_state: RSSMState,
        prev_action: torch.Tensor,
        embed: torch.Tensor,
    ) -> tuple[RSSMState, RSSMState]:
        prior = self.img_step(prev_state, prev_action)
        embed_hidden = torch.tanh(self.embed_layer(embed))
        x = torch.cat([prior.deter, embed_hidden], dim=-1)
        stats = self.posterior_layer(x)
        mean, std = stats.chunk(2, dim=-1)
        std = F.softplus(std) + self.min_std
        stoch = mean + std * torch.randn_like(mean)
        post = RSSMState(mean=mean, std=std, stoch=stoch, deter=prior.deter)
        return post, prior


class DreamerWorldModel(nn.Module):
    """DreamerV3-style latent world model tailored for GridLife."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: Optional[Dict] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        config = config or {}
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device or torch.device("cpu")

        embed_dim = config.get("embed_dim", 128)
        deter_dim = config.get("deter_dim", 200)
        stoch_dim = config.get("stoch_dim", 32)
        rssm_hidden = config.get("rssm_hidden", 200)
        model_hidden = config.get("model_hidden", 256)
        activation = config.get("activation", "elu")
        self.kl_scale = float(config.get("kl_scale", 1.0))
        self.free_nats = float(config.get("free_nats", 1.0))
        self.gamma = float(config.get("discount", 0.99))
        self.grad_clip = float(config.get("grad_clip", 1000.0))
        self.reward_scale = float(config.get("reward_scale", 1.0))
        self.value_scale = float(config.get("value_scale", 0.5))
        self.rollout_horizon = int(config.get("shield_horizon", 8))

        if activation == "relu":
            act_layer = nn.ReLU
        elif activation == "tanh":
            act_layer = nn.Tanh
        else:
            act_layer = nn.ELU

        def mlp(in_dim: int, out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, model_hidden),
                act_layer(),
                nn.Linear(model_hidden, model_hidden),
                act_layer(),
                nn.Linear(model_hidden, out_dim),
            )

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, model_hidden),
            act_layer(),
            nn.Linear(model_hidden, embed_dim),
        )

        self.decoder = mlp(deter_dim + stoch_dim, obs_dim)
        self.reward_predictor = mlp(deter_dim + stoch_dim, 1)
        self.value_predictor = mlp(deter_dim + stoch_dim, 1)

        self.rssm = RSSM(
            action_dim=action_dim,
            embed_dim=embed_dim,
            stoch_dim=stoch_dim,
            deter_dim=deter_dim,
            hidden_dim=rssm_hidden,
            min_std=float(config.get("min_std", 0.1)),
        )

        model_lr = float(config.get("model_lr", 3e-4))
        reward_lr = float(config.get("reward_lr", 3e-4))
        value_lr = float(config.get("value_lr", 3e-4))

        self.model_optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.rssm.parameters()),
            lr=model_lr,
        )
        self.reward_optimizer = torch.optim.Adam(
            self.reward_predictor.parameters(), lr=reward_lr
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_predictor.parameters(), lr=value_lr
        )

        self.is_dreamer = True
        self.supports_imagination = True
        self.to(self.device)

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _stack_states(self, states: Iterable[RSSMState]) -> Dict[str, torch.Tensor]:
        stacked = {
            "mean": torch.stack([s.mean for s in states], dim=1),
            "std": torch.stack([s.std for s in states], dim=1),
            "stoch": torch.stack([s.stoch for s in states], dim=1),
            "deter": torch.stack([s.deter for s in states], dim=1),
        }
        return stacked

    def observe(
        self,
        embeds: torch.Tensor,
        actions: torch.Tensor,
        nonterms: Optional[torch.Tensor] = None,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        batch_size, time_steps, _ = embeds.shape
        state = self.rssm.init_state(batch_size, embeds.device)
        posts: list[RSSMState] = []
        priors: list[RSSMState] = []
        prev_action = torch.zeros(batch_size, self.action_dim, device=embeds.device)
        for t in range(time_steps):
            action_t = actions[:, t]
            embed_t = embeds[:, t]
            post, prior = self.rssm.obs_step(state, prev_action, embed_t)
            if nonterms is not None:
                mask = nonterms[:, t].unsqueeze(-1)
                post = RSSMState(
                    mean=mask * post.mean + (1 - mask) * state.mean,
                    std=mask * post.std + (1 - mask) * state.std,
                    stoch=mask * post.stoch + (1 - mask) * state.stoch,
                    deter=mask * post.deter + (1 - mask) * state.deter,
                )
                prior = RSSMState(
                    mean=mask * prior.mean + (1 - mask) * state.mean,
                    std=mask * prior.std + (1 - mask) * state.std,
                    stoch=mask * prior.stoch + (1 - mask) * state.stoch,
                    deter=mask * prior.deter + (1 - mask) * state.deter,
                )
            posts.append(post)
            priors.append(prior)
            state = post
            prev_action = action_t
        return self._stack_states(posts), self._stack_states(priors)

    def imagine(
        self, actions: torch.Tensor, start_state: Optional[RSSMState] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size = actions.size(0)
        device = actions.device
        state = start_state or self.rssm.init_state(batch_size, device)
        states: list[RSSMState] = []
        for t in range(actions.size(1)):
            action_t = actions[:, t]
            state = self.rssm.img_step(state, action_t)
            states.append(state)
        return self._stack_states(states)

    def get_features(self, stacked_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([stacked_state["stoch"], stacked_state["deter"]], dim=-1)

    # ------------------------------------------------------------------
    # Training & evaluation utilities
    # ------------------------------------------------------------------
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs_seq"].to(self.device)
        actions = batch["action_seq"].to(self.device)
        rewards = batch["reward_seq"].to(self.device)
        nonterms = 1.0 - batch.get("done_seq", torch.zeros_like(rewards)).to(
            self.device
        )

        embeds = self.encoder(obs.view(-1, self.obs_dim)).view(
            obs.size(0), obs.size(1), -1
        )
        posts, priors = self.observe(embeds, actions, nonterms)
        feats = self.get_features(posts)

        recon = self.decoder(feats.view(-1, feats.size(-1))).view_as(obs)
        recon_loss = F.mse_loss(recon, obs)

        kl_values = _gaussian_kl(
            posts["mean"], posts["std"], priors["mean"], priors["std"]
        )
        if self.free_nats > 0.0:
            kl_values = torch.clamp(kl_values, min=self.free_nats)
        kl_loss = kl_values.mean()

        reward_pred = self.reward_predictor(feats.view(-1, feats.size(-1))).view_as(
            rewards
        )
        reward_loss = F.mse_loss(reward_pred, rewards)

        returns = torch.zeros_like(rewards)
        next_return = torch.zeros(rewards.size(0), device=self.device)
        for t in reversed(range(rewards.size(1))):
            next_return = rewards[:, t] + self.gamma * nonterms[:, t] * next_return
            returns[:, t] = next_return
        value_pred = self.value_predictor(feats.view(-1, feats.size(-1))).view_as(
            rewards
        )
        value_loss = F.mse_loss(value_pred, returns.detach())

        total_loss = (
            recon_loss
            + self.kl_scale * kl_loss
            + self.reward_scale * reward_loss
            + self.value_scale * value_loss
        )

        self.model_optimizer.zero_grad()
        self.reward_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.model_optimizer.step()
        self.reward_optimizer.step()
        self.value_optimizer.step()

        return {
            "recon_loss": float(recon_loss.detach().cpu()),
            "kl_loss": float(kl_loss.detach().cpu()),
            "reward_loss": float(reward_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "total_loss": float(total_loss.detach().cpu()),
        }

    @torch.no_grad()
    def infer_posterior(
        self, obs: torch.Tensor, prev_action: Optional[torch.Tensor] = None
    ) -> RSSMState:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        prev_action = (
            prev_action.to(self.device)
            if prev_action is not None
            else torch.zeros(obs.size(0), self.action_dim, device=self.device)
        )
        embed = self.encoder(obs)
        init_state = self.rssm.init_state(obs.size(0), self.device)
        post, _ = self.rssm.obs_step(init_state, prev_action, embed)
        return post

    @torch.no_grad()
    def imagine_rollout(
        self,
        initial_obs: torch.Tensor,
        action_sequences: torch.Tensor,
        internal_model,
        viability_approximator,
        initial_internal_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if action_sequences.dim() == 2:
            action_sequences = action_sequences.unsqueeze(0)
        action_sequences = action_sequences.to(self.device)
        num_candidates, horizon, _ = action_sequences.shape

        initial_obs = initial_obs.to(self.device)
        if initial_obs.dim() == 1:
            initial_obs = initial_obs.unsqueeze(0)
        posterior = self.infer_posterior(initial_obs)
        start_state = RSSMState(
            mean=posterior.mean.repeat(num_candidates, 1),
            std=posterior.std.repeat(num_candidates, 1),
            stoch=posterior.stoch.repeat(num_candidates, 1),
            deter=posterior.deter.repeat(num_candidates, 1),
        )
        imagined = self.imagine(action_sequences, start_state)
        feats = self.get_features(imagined)
        pred_obs = self.decoder(feats.view(-1, feats.size(-1))).view(
            num_candidates, horizon, self.obs_dim
        )

        if initial_internal_state is None:
            raise ValueError(
                "DreamerWorldModel.imagine_rollout requires an initial internal state."
            )

        internal_state = torch.as_tensor(
            initial_internal_state, dtype=torch.float32, device=self.device
        )
        if internal_state.dim() == 1:
            internal_state = internal_state.unsqueeze(0)
        if internal_state.size(0) == 1:
            internal_state = internal_state.repeat(num_candidates, 1)

        predicted_internal: list[torch.Tensor] = []
        margins: list[torch.Tensor] = []
        state = internal_state
        for t in range(horizon):
            action_t = action_sequences[:, t]
            state = internal_model.predict_next(state, action_t)
            predicted_internal.append(state)
            margin = viability_approximator.get_margin(state)
            if margin.dim() == 1:
                margin = margin.unsqueeze(-1)
            margins.append(margin.squeeze(-1))

        predicted_internal_tensor = torch.stack(predicted_internal, dim=1)
        margins_tensor = torch.stack(margins, dim=1)

        return {
            "predicted_observations": pred_obs,
            "predicted_internal_states": predicted_internal_tensor,
            "margins": margins_tensor,
        }

    @torch.no_grad()
    def rollout_viability(
        self,
        initial_obs,
        initial_internal_state,
        action_sequences,
        internal_model,
        viability_approximator,
    ) -> Dict[str, torch.Tensor]:
        initial_obs_tensor = torch.as_tensor(
            initial_obs, dtype=torch.float32, device=self.device
        )
        initial_internal_tensor = torch.as_tensor(
            initial_internal_state, dtype=torch.float32, device=self.device
        )
        action_seq_tensor = torch.as_tensor(
            action_sequences, dtype=torch.float32, device=self.device
        )
        return self.imagine_rollout(
            initial_obs_tensor,
            action_seq_tensor,
            internal_model,
            viability_approximator,
            initial_internal_state=initial_internal_tensor,
        )

    @torch.no_grad()
    def compute_surprise_reward(self, obs, act, next_obs) -> float:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        next_obs_tensor = torch.as_tensor(
            next_obs, dtype=torch.float32, device=self.device
        )
        act_tensor = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        posterior = self.infer_posterior(obs_tensor)
        prior = self.rssm.img_step(
            posterior, act_tensor.unsqueeze(0) if act_tensor.dim() == 1 else act_tensor
        )
        recon = self.decoder(self.rssm.get_feat(prior))
        if recon.dim() == 2:
            recon = recon.squeeze(0)
        loss = F.mse_loss(recon, next_obs_tensor)
        return float(loss.detach().cpu())

    def train_model(self, *args, **kwargs):
        raise NotImplementedError(
            "DreamerWorldModel expects train_step() with sequence batches."
        )
