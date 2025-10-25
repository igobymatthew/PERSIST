# PERSIST

A persistence-first reinforcement learning stack that combines homeostatic control, safety shielding, and intrinsic drives so agents can stay viable while pursuing long-horizon goals.

## Overview
- **Purpose**: Provide a batteries-included reference implementation for persistence-centric research and experiments.
- **What you get**: A guided CLI, modular safety and intrinsic-motivation components, and ready-to-run scenarios for single- and multi-agent studies.
- **How to explore**: Start with the quickstart below, then jump into the configuration map to toggle the modules you need.

## Quickstart
### 1. Environment setup
It is recommended to use Conda to manage dependencies.

1. **Create the Conda environment**
   ```bash
   conda env create -f environment.yml
   ```
2. **Activate the environment**
   ```bash
   conda activate persist
   ```

### 2. Launch the guided CLI
From the project root run:

```bash
python main.py
```

The CLI walks through the standard single-agent build, optional multi-agent extensions, robustness toggles, and a "run from config" shortcut. You will see a confirmation card summarizing the final configuration before training starts.

### 3. Continue learning
- Follow the [hands-on walkthroughs](docs/use_case_walkthroughs.md) for baseline, multi-agent, and adversarial scenarios.
- Review [training resilience tactics](docs/training_resilience.md) when designing custom learning loops.

## Configuration map
The `config.yaml` file wires every subsystem. Use the tables below to see where the implementations live and which knobs are considered core versus optional.

### Core subsystems
| Capability | Modules & docs | Location |
| --- | --- | --- |
| Homeostasis & viability | Homeostat, ViabilityApproximator, Safety Shield · [theory primer](docs/theory.md) | `components/`, `systems/`, `schemas/` |
| Policy & training loop | PersistAgent, SAC/CTDE trainers, replay buffers | `agents/`, `utils/trainer.py`, `buffers/` |
| World modelling & intrinsic drives | Latent/Dreamer world models, empowerment, surprise, RND | `components/latent_world_model.py`, `components/dreamer_world_model.py`, `components/empowerment.py`, `components/rnd.py` |
| Configuration schema & validation | JSON/YAML schemas, CLI | `schemas/`, `config.yaml`, `main.py` |

### Optional extensions
| Capability | Modules & docs | Location |
| --- | --- | --- |
| Multi-agent coordination | Cooperative/competitive CTDE stack, Biodiversity Fabric Simulator · [deep dive](docs/multiagent.md) | `multiagent/`, `population/`, `environments/` |
| Emotional Equilibrium Atlas++ | Affect-aware agents and life-stage choreography · [doc](docs/EEA.md) | `agents/eea_agent.py`, `population/`, `systems/` |
| Growth-mimetic roadmap | Stage previews, lineage memory, biodiversity simulators · [technology brief](docs/growth_mimetic_technologies.md) | `evolution/`, `population/`, `docs/` |
| Telemetry & operations | Prometheus telemetry, alerts, reporting | `ops/`, `utils/reporting.py` |
| Robustness & adversaries | PGD adversary, safety probes, OOD detectors | `components/adversary.py`, `components/safety_probe.py`, `components/ood_detector.py` |
| LoRA inference tooling | Adapter-aware model loader | `tools/lora_inference.py`, `tools/__init__.py` |

## Key modules at a glance
- **Experiment Coordinator** (`systems/coordinator.py`): orchestrates environments, agents, and persistence subsystems.
- **ComponentFactory** (`utils/factory.py`): builds agents, shields, and optional modules from `config.yaml`.
- **PersistenceManager** (`systems/persistence.py`): handles checkpointing, degraded-mode fallbacks, and restart flows.
- **TelemetryManager** (`ops/telemetry.py`): exposes survival, constraint, and entropy metrics via Prometheus.
- **Dreamer World Model** (`components/dreamer_world_model.py`): unlocks imagination rollouts for intrinsic planning.

## Growth-mimetic roadmap
The roadmap documents how growth-inspired systems land in the stack:
- [docs/roadmap.md](docs/roadmap.md) for milestone tracking and verification plans.
- [docs/growth_mimetic_devlog/](docs/growth_mimetic_devlog) for implementation notes.
- [docs/growth_mimetic_technologies.md](docs/growth_mimetic_technologies.md) for vision statements and future modules.

## Deep-dive references
- [docs/theory.md](docs/theory.md): mathematical derivations, philosophical framing, and reward formulations.
- [docs/multiagent.md](docs/multiagent.md): cooperative vs. competitive flows, BFS scenarios, and lineage tie-ins.
- [docs/EEA.md](docs/EEA.md): affective agents and life-stage integrations.
- [docs/summary.md](docs/summary.md): narrative overview of the persistence stack.
- [docs/changelog.md](docs/changelog.md): full historical progress timeline.

### Progress Updates
* **2025-10-27T16:30:00+00:00**: Modularized onboarding docs and relocated deep dives.
  * Rebuilt the README around overview, quickstart, and configuration maps so newcomers can stand up experiments without sifting through theoretical derivations.
  * Added `docs/changelog.md` for the full progress journal and `docs/multiagent.md` for cooperative/competitive walkthroughs, keeping deep content a click away while preserving navigability.
* _Earlier updates now live in [docs/changelog.md](docs/changelog.md)._ 
