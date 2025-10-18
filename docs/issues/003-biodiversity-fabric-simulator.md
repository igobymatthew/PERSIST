# Issue: Build the Biodiversity Fabric Simulator

## Summary
Milestone M3 in `docs/growth_mimetic_technologies.md` calls for a Biodiversity Fabric Simulator (BFS) that introduces species archetypes, habitat succession knobs, and richness telemetry. The current environments (`environments/grid_life.py`, `environments/multi_agent_gridlife.py`) do not yet model species diversity or ecological succession. This issue delivers the BFS layer.

## Implementation Plan
1. **Species archetype system**
   - Design a configuration schema (extend `schemas/viability.schema.json` or add `schemas/biodiversity.schema.json`) that defines species-specific metabolism, constraints, and maintenance tasks.
   - Implement archetype loaders in `utils/factory.py` that instantiate agents with species-specific parameters and shield settings.
2. **Habitat succession mechanics**
   - Extend the curriculum or environment generator to evolve hazards, resources, and adversaries over time (e.g., via genetic algorithms as described in the doc).
   - Introduce hooks in `environments/multi_agent_gridlife.py` to apply succession updates at configurable intervals.
3. **Population-level safety & telemetry**
   - Create an `EnsembleShield` or coordinator module that evaluates viability at the population level (respecting inter-species constraints).
   - Add Prometheus metrics for species richness, trophic stability, and mutualism counters in `ops/telemetry.py`.
4. **Testing & documentation**
   - Add smoke tests in `tests/` that spawn multiple species and verify no catastrophic collapse occurs under nominal settings.
   - Document setup and configuration in `docs/growth_mimetic_technologies.md` and add a walkthrough to `docs/use_case_walkthroughs.md`.

## Acceptance Criteria
- Configuration supports defining multiple species with unique constraints and reward weights.
- Environments evolve resources and hazards over time according to succession rules.
- Telemetry captures richness/stability metrics and exposes them through Prometheus.
- Regression tests cover multi-species episodes and succession transitions.
- Documentation guides users through enabling BFS scenarios.
