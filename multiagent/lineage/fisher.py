"""Fisher information estimation helpers for TMW."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F


def _zero_like_named_parameters(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: torch.zeros_like(param, device=param.device)
        for name, param in module.named_parameters()
        if param.requires_grad
    }


def _normalize_fisher(
    fisher: Dict[str, torch.Tensor], *, epsilon: float = 1e-8
) -> Dict[str, torch.Tensor]:
    normalized: Dict[str, torch.Tensor] = {}
    for name, tensor in fisher.items():
        if tensor.numel() == 0:
            normalized[name] = tensor
            continue
        scale = tensor.abs().max().clamp_min(epsilon)
        normalized[name] = tensor / scale
    return normalized


def estimate_actor_fisher(
    actor: torch.nn.Module,
    batches: Iterable[torch.Tensor],
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Estimate diagonal Fisher information for a stochastic actor."""

    actor.eval()
    accum = _zero_like_named_parameters(actor)
    total = 0

    for obs_batch in batches:
        total += 1
        obs = obs_batch.to(device or next(actor.parameters()).device)
        actor.zero_grad(set_to_none=True)
        _, logp = actor(obs, deterministic=False, with_logprob=True)
        # Negative log likelihood; fisher uses gradient of log prob.
        loss = -logp.mean()
        loss.backward()
        for name, param in actor.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            accum[name] += param.grad.detach() ** 2

    if total == 0:
        return {name: tensor.detach().cpu() for name, tensor in accum.items()}

    fisher = {name: (tensor / total).detach() for name, tensor in accum.items()}
    return {name: value.cpu() for name, value in _normalize_fisher(fisher).items()}


def estimate_viability_fisher(
    viability_model: torch.nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Estimate Fisher information for the viability approximator."""

    viability_model.eval()
    accum = _zero_like_named_parameters(viability_model)
    total = 0

    for states, labels in batches:
        total += 1
        x = states.to(device or next(viability_model.parameters()).device)
        y = labels.to(x.device)
        viability_model.zero_grad(set_to_none=True)
        logits = viability_model(x)
        logits = logits.view_as(y)
        loss = F.binary_cross_entropy(logits, y, reduction="mean")
        loss.backward()
        for name, param in viability_model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            accum[name] += param.grad.detach() ** 2

    if total == 0:
        return {name: tensor.detach().cpu() for name, tensor in accum.items()}

    fisher = {name: (tensor / total).detach() for name, tensor in accum.items()}
    return {name: value.cpu() for name, value in _normalize_fisher(fisher).items()}
