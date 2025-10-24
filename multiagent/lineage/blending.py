"""Fisher-masked blending routines for TMW."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch

from .archive import LineageRecord


@dataclass
class BlendConfig:
    """Runtime configuration for lineage blending."""

    alpha: float = 0.5
    lora_rank: int = 8
    max_ancestors: int = 3


class LineageBlender:
    """Compose agent parameters from lineage records using Fisher masks."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def blend_agent(
        self,
        *,
        agent,
        viability_model: Optional[torch.nn.Module],
        safety_network: Optional[torch.nn.Module],
        ancestors: Iterable[LineageRecord],
        config: BlendConfig,
    ) -> None:
        """Blend lineage knowledge directly into the provided agent."""

        ancestors = list(ancestors)
        if not ancestors:
            return

        policy_state = agent.get_state()
        blended_policy = self._blend_state_dict(
            base_state=policy_state,
            ancestors=[
                record.policy_state for record in ancestors if record.policy_state
            ],
            fisher_masks=[record.fisher_mask for record in ancestors],
            config=config,
        )
        agent.load_state(blended_policy)

        if hasattr(agent, "load_optimizer_state"):
            # Reset optimizers; new lineage should start with fresh momentum.
            agent.load_optimizer_state({})

        if viability_model is not None:
            viability_state = viability_model.state_dict()
            ancestor_viability = [
                record.viability_state
                for record in ancestors
                if record.viability_state is not None
            ]
            blended_viability = self._blend_state_dict(
                base_state=viability_state,
                ancestors=ancestor_viability,
                fisher_masks=[record.fisher_mask for record in ancestors],
                config=config,
            )
            viability_model.load_state_dict(blended_viability)

        if safety_network is not None:
            safety_state = safety_network.state_dict()
            ancestor_safety = [
                record.safety_state
                for record in ancestors
                if record.safety_state is not None
            ]
            if ancestor_safety:
                blended_safety = self._blend_state_dict(
                    base_state=safety_state,
                    ancestors=ancestor_safety,
                    fisher_masks=[record.fisher_mask for record in ancestors],
                    config=config,
                )
                safety_network.load_state_dict(blended_safety)

        affect_buffer = getattr(agent, "affect_buffer", None)
        if affect_buffer is not None:
            for record in ancestors:
                if record.affect_state is None:
                    continue
                self._load_affect_state(affect_buffer, record.affect_state)
                break  # Only hydrate from the most recent compatible snapshot.

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _blend_state_dict(
        self,
        *,
        base_state: Dict[str, torch.Tensor],
        ancestors: Iterable[Optional[Dict[str, torch.Tensor]]],
        fisher_masks: Iterable[Optional[Dict[str, torch.Tensor]]],
        config: BlendConfig,
    ) -> Dict[str, torch.Tensor]:
        state = {name: tensor.clone() for name, tensor in base_state.items()}
        alpha = float(config.alpha)
        rank = max(1, int(config.lora_rank))

        paired = [
            (ancestor, mask)
            for ancestor, mask in zip(ancestors, fisher_masks)
            if ancestor is not None
        ][: config.max_ancestors]

        if not paired:
            return state

        for ancestor_state, mask in paired:
            for name, tensor in ancestor_state.items():
                if name not in state:
                    continue
                ancestor_tensor = tensor.to(state[name].device)
                base_tensor = state[name]
                delta = self._compute_lora_delta(
                    base_tensor=base_tensor,
                    ancestor_tensor=ancestor_tensor,
                    rank=rank,
                )
                if mask and name in mask:
                    mask_tensor = mask[name].to(delta.device)
                    delta = delta * (1.0 / (1.0 + mask_tensor))
                state[name] = base_tensor + alpha * delta
        return state

    def _compute_lora_delta(
        self,
        *,
        base_tensor: torch.Tensor,
        ancestor_tensor: torch.Tensor,
        rank: int,
    ) -> torch.Tensor:
        diff = ancestor_tensor - base_tensor
        if diff.ndim < 2 or min(diff.shape) <= rank:
            return diff
        try:
            u, s, vh = torch.linalg.svd(diff, full_matrices=False)
        except RuntimeError:
            # Fall back to direct difference if SVD fails to converge.
            return diff
        effective_rank = min(rank, s.numel())
        if effective_rank == 0:
            return torch.zeros_like(diff)
        u_r = u[:, :effective_rank]
        s_r = s[:effective_rank]
        vh_r = vh[:effective_rank, :]
        return (u_r * s_r) @ vh_r

    def _load_affect_state(self, buffer, state: Dict[str, torch.Tensor]) -> None:
        if hasattr(buffer, "load_state_dict"):
            buffer.load_state_dict(state)
        elif hasattr(buffer, "deserialize"):
            buffer.deserialize(state)
        elif hasattr(buffer, "__setstate__"):
            buffer.__setstate__(state)
