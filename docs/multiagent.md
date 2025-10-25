# Multi-agent Coordination in PERSIST

This guide explains how the cooperative and competitive flows are composed when you enable the multi-agent stack. It complements the onboarding-oriented README by diving into the centralized-training, decentralized-execution (CTDE) architecture and the Biodiversity Fabric Simulator (BFS).

## Architecture overview
- **Centralized training**: `multiagent/trainer.py` coordinates gradient updates for the shared policies, replay buffers, and viability ensembles.
- **Decentralized execution**: `multiagent/policies/` exposes per-agent wrappers so each species or role acts autonomously at inference time.
- **Population governance**: `population/ensemble_shield.py`, `population/lineage/`, and BFS instrumentation maintain viability votes, lineage blending, and environment balance across species.
- **Experiment control**: `ComponentFactory` and `ExperimentCoordinator` wire the above modules using the `multiagent` and `population` sections from `config.yaml`.

## Cooperative flows
1. **Role-aware policy sharing**: `multiagent/shared_sac.py` shares actor/critic weights while injecting role embeddings so different agent archetypes can coordinate without duplicating networks.
2. **Shared replay buffers**: `buffers/replay_ma.py` stores joint transitions and role metadata, unlocking centralized updates and near-boundary sampling for the safety ensemble.
3. **Viability consensus**: The `EnsembleShield` aggregates safety votes from species-specific viability approximators before projecting actions back to each agent.
4. **Life-stage synchronization**: When `LifeStageManager` is enabled, the trainer injects life-stage context into both cooperative policy updates and EEA affectors so agents mature together.

## Competitive and adversarial hooks
- **Adversarial perturbations**: Enabling the `adversarial` block in `config.yaml` adds PGD perturbations via `components/adversary.py`, stress-testing cooperative policies under targeted attacks.
- **Resource contention**: `components/budget_meter.py` and the maintenance tasks in `environments/grid_life.py` enforce shared energy, cooldown, and repair budgets, forcing agents to negotiate scarce resources.
- **Shield veto escalation**: Configure `viability_ensemble.voting` to `"veto_if_any_unsafe"` when you want safety votes from risk-sensitive agents to override aggressive teammates.

## Biodiversity Fabric Simulator (BFS)
The BFS extends the base multi-agent grid into trophic layers and regeneration cycles.

- **Species archetypes**: Define species templates under `population/species/` and reference them via `population.species_catalog` in `config.yaml`.
- **Succession knobs**: Use `population.successions` to choreograph how species spawn, respawn, and seed lineage memories across life stages.
- **Telemetry**: BFS scenarios emit richness, trophic stability, and mutualism metrics via the Prometheus gauges registered in `ops/telemetry.py`.

## Configuration quick reference
| Key | Purpose |
| --- | --- |
| `multiagent.enabled` | Toggles centralized multi-agent training. |
| `multiagent.trainer` | Selects the trainer (e.g., `shared_sac`). |
| `multiagent.roles` | Lists role identifiers and embeddings for each species or policy head. |
| `population.ensemble_shield` | Configures how safety votes are aggregated. |
| `population.lineage` | Enables lineage memory, respawn initialization, and Fisher-masked blending. |
| `population.fire_cycle` | Controls regeneration cadence for the fire-reset philosophy. |

## Further reading
- [docs/use_case_walkthroughs.md](docs/use_case_walkthroughs.md): step-by-step BFS and multi-agent exercises.
- [docs/EEA.md](docs/EEA.md): how affective agents integrate into lineage-aware multi-agent runs.
- [docs/growth_mimetic_technologies.md](docs/growth_mimetic_technologies.md): roadmap context for biodiversity and lineage mechanics.
