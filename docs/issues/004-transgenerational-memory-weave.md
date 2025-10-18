# Issue: Implement the Transgenerational Memory Weave

## Summary
Milestone M4 in `docs/growth_mimetic_technologies.md` outlines a Transgenerational Memory Weave (TMW) where policies, viability approximators, and affect priors propagate across lifespans using Fisher-masked blending. The current persistence stack checkpoints agents (`systems/coordinator.py`) but does not yet archive or blend lineage knowledge. This issue scopes the work to deliver TMW.

## Implementation Plan
1. **Lineage archive subsystem**
   - Create a `lineage` module (e.g., `multiagent/lineage/`) with utilities to snapshot policy weights, viability models, and affect buffers at stage completions or death events.
   - Store metadata (stage, environment seed, metrics) alongside checkpoints for downstream selection.
2. **Fisher-masked blending**
   - Implement Fisher Information estimation for policy networks (reuse experience batches from the replay buffer) and store masks per ancestor.
   - Add blending routines that initialize new agents by interpolating ancestor weights with LoRA adapters while respecting Fisher masks.
3. **Spawn-time initialization**
   - Extend the factory/coordinator so when an agent respawns, it loads the lineage archive, selects relevant ancestors (e.g., same species), and composes the new policy + viability approximator.
   - Integrate with the existing `SafetyNetwork` so amortized shields benefit from inherited experience.
4. **Evaluation and tooling**
   - Build benchmarks in `benchmarks/` that measure recovery speed and unsafe exploration rates with/without TMW.
   - Add CLI commands (e.g., `python main.py --lineage-report`) to inspect lineage graphs.
   - Document workflows in `docs/growth_mimetic_technologies.md` and start a `docs/growth_mimetic_devlog/` entry summarizing lineage experiments.

## Acceptance Criteria
- Lineage archives capture policies/approximators with metadata at the end of each life stage or death event.
- New agents can initialize from archived knowledge while honoring Fisher masks and LoRA adapters.
- Benchmarks demonstrate improved recovery and reduced unsafe exploration compared with a baseline.
- CLI tooling exposes lineage status for debugging.
- Documentation reflects TMW behavior and configuration knobs.
