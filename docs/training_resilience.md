# Training for Resilient Persistence

Persistent agents in PERSIST must continue learning after disruptive "fire" events while remaining stable during long stretches of autonomous operation. This guide summarizes optimizer and parameter-efficient fine-tuning (PEFT) strategies that support those goals.

## Optimizer Recommendations

Fire events can abruptly perturb model weights, so optimizers need to combine stability with responsiveness to sparse updates.

| Optimizer | Why it Helps Persistence |
| --- | --- |
| **AdamW** | Reliable baseline whose decoupled weight decay improves continual-learning stability. |
| **RAdam** | Rectified variance adapts the early training steps, stabilizing recovery after a fire-induced shock. |
| **Lookahead + RAdam** | Lookahead smooths the weight trajectory while RAdam keeps steps well-scaled, balancing exploration and robustness. |
| **Lion** | Momentum-based update rule with strong generalization for transformer-heavy policies and world models. |
| **SAM (Sharpness-Aware Minimization)** | Seeks flat minima, yielding policies that rebound quickly after resets. |
| **RMSprop / Adafactor** | Lightweight choices for multi-agent deployments where compute and memory are constrained. |

## Parameter-Efficient Fine-Tuning (LoRA) Options

Fire-triggered resets are easier to handle when plasticity is concentrated in modular adapters rather than the full backbone. The following frameworks support that pattern:

| Framework | Key Benefits |
| --- | --- |
| **PEFT (Hugging Face)** | Straightforward LoRA and QLoRA integration for PyTorch models—freeze most weights and adapt selectively. |
| **DeepSpeed-LoRA** | Scales to large, distributed training runs with mixed precision, ideal for multi-agent curricula. |
| **LoRA + EWC Hybrid** | Pair fast-learning LoRA adapters with Elastic Weight Consolidation on the backbone to preserve long-term memory. |
| **DoRA (Decoupled LoRA)** | Decouples direction and magnitude updates for more predictable gradients in safety-critical subsystems. |

## Integrating with the Fire Philosophy

- **Optimizer schedules** can restart or anneal after each fire to encourage rapid recovery without destabilizing long-term knowledge.
- **Adapters** become the evolutionary substrate: swap in new LoRA modules, archive or compress stale ones, and keep the backbone intact.
- **Curriculum design** can align fire frequency with exploration phases—RAdam or SAM handle high-volatility intervals, while Lookahead maintains steady progress between shocks.

By pairing resilient optimizers with modular PEFT components, fire events become controlled opportunities for adaptation rather than catastrophic setbacks.

