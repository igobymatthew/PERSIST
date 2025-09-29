import logging
from typing import Optional

import torch
import torch.nn as nn


class FireEvent:
    """Utility for applying rapid plasticity changes after a fire trigger."""

    event_log = []

    @classmethod
    def apply(
        cls,
        model: Optional[nn.Module],
        prune_fraction: float = 0.1,
        threshold_scale: float = 0.5,
    ) -> None:
        """Apply pruning and threshold mutation to the provided model."""
        if model is None:
            return

        logger = logging.getLogger("FireEvent")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        logger.info("🔥 FireEvent triggered: compressing weights and lowering thresholds.")
        cls._prune_weights(model, prune_fraction, logger)
        cls._mutate_thresholds(model, threshold_scale, logger)

        cls.event_log.append(
            {
                "message": "FireEvent applied",
                "prune_fraction": prune_fraction,
                "threshold_scale": threshold_scale,
            }
        )

    @staticmethod
    def _prune_weights(model: nn.Module, prune_fraction: float, logger: logging.Logger) -> None:
        if prune_fraction <= 0:
            return

        for name, param in model.named_parameters():
            if param.requires_grad and param.dim() >= 2:
                flat_abs = param.data.abs().flatten()
                if flat_abs.numel() == 0:
                    continue
                k = max(1, int(prune_fraction * flat_abs.numel()))
                if k > flat_abs.numel():
                    k = flat_abs.numel()
                threshold, _ = torch.kthvalue(flat_abs, k)
                mask = param.data.abs() <= threshold
                pruned = mask.sum().item()
                if pruned > 0:
                    param.data[mask] = 0.0
                    logger.info(
                        "Pruned %d weights below %.4f in parameter %s",
                        pruned,
                        threshold.item() if threshold.numel() else float("nan"),
                        name,
                    )

    @staticmethod
    def _mutate_thresholds(
        model: nn.Module, threshold_scale: float, logger: logging.Logger
    ) -> None:
        if threshold_scale == 1.0:
            return

        for module in model.modules():
            if hasattr(module, "threshold"):
                threshold_tensor = module.threshold
                if isinstance(threshold_tensor, nn.Parameter):
                    new_value = threshold_tensor.data * threshold_scale
                    module.threshold.data.copy_(new_value)
                elif isinstance(threshold_tensor, torch.Tensor):
                    module.threshold.mul_(threshold_scale)
                else:
                    continue
                logger.info(
                    "Scaled threshold for %s by %.3f",
                    module.__class__.__name__,
                    threshold_scale,
                )

    @classmethod
    def get_event_log(cls):
        return list(cls.event_log)
