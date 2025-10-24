# Growth-Mimetic Technology Pitch

## 1. Vision Alignment
PERSIST already treats survival as a synthesis of homeostasis, viability shielding, and intrinsic curiosity, which mirrors how living systems sustain themselves across changing environments. Building on that foundation, these proposals extend persistence into developmental intelligence: agents will now experience age-specific growth arcs, emotional maturation, and biodiversity-aware interactions that evolve alongside the world they inhabit.

## 2. LifeStage Dynamics Graph (LDG)
**Goal:** Encode biological growth stages as structured trajectories that modulate metabolism, cognition, and risk tolerance.

- **Developmental phases:** Define discrete life stages (infant, juvenile, adult, elder) with their own homeostatic setpoints and viability thresholds. Curriculum logic gradually tightens constraints, echoing how metabolic and social expectations escalate with age.
- **Stage transition triggers:** Couple stage shifts to cumulative survival time, shield interventions, and internal energy reserves. Each transition re-parameterizes the shield, maintenance tasks, and budget meters so resource priorities change as the agent matures.
- **Implementation hooks:**
  - Extend the viability schema with age-indexed constraint bands and connect them to the existing shield projection flow.
  - Use the NearBoundaryBuffer to emphasize experiences near developmental tipping points.
  - Schedule optimizer resets (fire events) at stage changes to emulate growth spurts, combining Lookahead+RAdam with LoRA adapters for fast recalibration.
  - **Status:** Schema wiring, stage telemetry, and fire-event resets landed in commit `8638d78` on 2025-10-16 (`schemas/viability.schema.json`, `components/life_stage.py`, `main.py`, `systems/coordinator.py`). Shield parameter tuning remains open alongside the Dreamer rollout (see `docs/issues/001-dreamer-world-model-upgrade.md`).

## 3. Emotional Equilibrium Atlas++ (EEA++)
**Goal:** Transform the Emotional Equilibrium Architecture into a longitudinal affective growth model.

- **Affective scaffolding:** Track happiness and fear ratios per stage, enforcing healthy ranges that drift toward adult equilibria. Entropy buffers expand in adolescence to encourage exploration, then narrow in adulthood to stabilize.
- **Contextual memories:** Introduce a context-weighted experience bank that distills salient emotional episodes. During fire events, replay the bank to rebuild balanced affective policies faster.
- **Telemetric storytelling:** Expose stage-aware affect metrics through the TelemetryManager so downstream dashboards narrate emotional maturation over time.
  - **Status:** Stage-aware targets, entropy reseeding, and telemetry payloads shipped across commits `b7a343b` (2025-10-19) and `42158dc` (2025-10-21). Follow `docs/issues/002-eea-plus-plus-stage-aware-affect.md` for remaining dashboard storytelling and replay analytics.

## 4. Biodiversity Fabric Simulator (BFS)
**Goal:** Achieve one-to-one ecological plausibility by letting agents co-evolve in ecosystems shaped by diversity rules.

- **Species archetypes:** Multi-agent roles now draw species-specific metabolism, maintenance preferences, and reward shaping from the declarative schema introduced in `schemas/biodiversity.schema.json`. `utils/factory.ComponentFactory.get_species_catalog()` exposes the archetype registry for downstream planners, and `environments/multi_agent_gridlife.py` assigns species identities to each agent at reset.
- **Habitat succession:** The multi-agent GridLife environment rotates through configurable succession phases, dynamically adjusting food density, hazards, and disturbance intensity. Phases are defined in configuration under `biodiversity.succession` and applied via `_advance_succession_phase()`/`_regenerate_resources()` hooks.
- **Population persistence:** The new `PopulationSafetyCoordinator` (see `population/ensemble_shield.py`) evaluates species-level viability, surfacing richness, trophic stability, and mutualism counters on every step. `ops/telemetry.TelemetryManager.update_biodiversity()` streams those metrics to Prometheus so dashboards can track ecosystem health in real time.
  - **Status:** ✅ Implemented. Core simulator and archetypes arrived with `7237644` (2025-09-27); Prometheus gauges for richness and trophic stability were added in `e7a790f` (2025-10-21). Follow-up tuning tasks remain in `docs/issues/003-biodiversity-fabric-simulator.md` for curriculum automation.

## 5. Transgenerational Memory Weave (TMW)
**Goal:** Preserve and adapt knowledge across lifespans, imitating cultural transmission.

- **Evolutionary replay:** After each life stage or death event, archive distilled policies and viability approximators. Genetic operators mutate these archives while respecting Fisher-informed masks so important memories persist.
- **Heritable schemas:** When a new agent spawns, initialize it with stage-specific priors drawn from ancestors, blending imitation learning with shield amortization to reduce unsafe exploration.
- **Long-horizon evaluation:** Use benchmark suites to score lineage resilience, focusing on recovery rate after fire events and biodiversity impacts of inherited behaviors.
  - **Status:** ✅ Implemented via `multiagent/lineage/` utilities, lineage-enabled coordinator hooks, and CLI reporting in `main.py`. Benchmarks and devlog notes cover recovery/unsafe exploration deltas; follow-up tuning remains in `docs/issues/004-transgenerational-memory-weave.md`.

## 6. Next Steps
1. Prototype age-indexed viability schemas and integrate them into the CLI so users can configure developmental arcs without manual YAML edits.
2. Extend telemetry to log life-stage transitions, affect ratios, and species richness for experiment reproducibility.
3. Pair NSGA-II curriculum search with Transgenerational Memory Weave objectives to auto-design worlds that produce sustainable, diverse populations.

## 7. Implementation Roadmap (v0.1)

| Milestone | Scope | Primary Modules | Validation Signals |
|-----------|-------|-----------------|--------------------|
| **M0 – Schema & Telemetry Scaffolding** | Introduce life-stage fields to viability schemas and expose stage-aware metrics through the TelemetryManager. | `schemas/viability.py`, `components/telemetry/manager.py`, CLI config loaders. | ✅ Schema validation passes, ✅ CLI prints stage timeline preview, ✅ Telemetry logs include `life_stage` and `stage_age`. *(Status: ✅ Completed in commit `8638d78` on 2025-10-16.)*
| **M1 – LifeStage Dynamics Graph** | Implement age-indexed constraint bands and optimizer resets tied to stage transitions. | `components/viability`, `components/shield`, optimizer orchestration utilities. | ✅ Deterministic stage transitions in unit tests, ✅ Shield re-parameterizes budgets per stage, ✅ Optimizer reset triggers recorded in telemetry.
| **M2 – EEA++ Affect Loop** | Extend EEA buffers and replay hooks with stage-aware affect setpoints. | `agents/eea`, `buffers/emotion.py`. | ✅ Affect ratios respect stage-dependent bounds in simulations, ✅ Context bank replay shortens recovery after fire events. *(Status: ✅ Implemented across commits `b7a343b` (2025-10-19) and `42158dc` (2025-10-21); telemetry examples still to be documented.)*
| **M3 – Biodiversity Fabric Simulator** | Add species archetypes and habitat succession knobs to curriculum generators. | `schemas/biodiversity.schema.json`, `environments/multi_agent_gridlife.py`, `population/ensemble_shield.py`, `ops/telemetry.py`. | ✅ Multi-species smoke test passes with no catastrophic collapse, ✅ Telemetry exposes species richness trend. *(Status: ✅ Landed via `7237644` (2025-09-27) with telemetry uplift in `e7a790f` (2025-10-21); tuning in `docs/issues/003-biodiversity-fabric-simulator.md`.)*
| **M4 – Transgenerational Memory Weave** | Persist post-stage policies and inject them into new agents with Fisher-masked blending. | `multiagent/lineage`, `components/persistence`, `tools/checkpoints`. | ✅ Lineage benchmarks show faster recovery after fire events, ✅ Heritable priors reduce unsafe exploration in regression tests. *(Status: 🚧 Not started – see `docs/issues/004-transgenerational-memory-weave.md`.)*

## 8. Implementation Timeline Snapshots

- **2025-10-21 — `42158dc`**: FireEvent reseeding learned stage context, closing the EEA++ affect loop milestone.
- **2025-10-19 — `b7a343b`**: Emotional Equilibrium Agent pulled life-stage targets, emitted telemetry payloads, and seeded per-stage replay banks.
- **2025-10-18 — `3262acf`**: Dreamer RSSM backbone merged, enabling imagination rollouts to drive shield projections.
- **2025-10-16 — `8638d78`**: Life-stage schema, CLI preview, and telemetry gauges landed in one sweep.
- **2025-10-04 — `8c1e9f2`**: GA/NSGA-II engine became available through the CLI meta-optimization workflow.
- **2025-09-27 — `7237644`**: Biodiversity Fabric Simulator introduced species archetypes, succession knobs, and multi-agent safety coordination.
- **2025-09-27 — `1e843f6`**: Continual learning stack adopted Elastic Weight Consolidation with rehearsal buffers.

## 9. Documentation & Collaboration Hooks

- **Design logs:** Add a `docs/growth_mimetic_devlog/` directory to capture week-to-week decisions, data schema migrations, and telemetry screenshots once experiments begin.
- **Issue templates:** Create GitHub issue templates for each milestone (Schema, LDG, EEA++, BFS, TMW) so contributors can self-assign subtasks and reference this pitch directly.
- **Cross-references:** Update `docs/EEA.md`, `docs/use_case_walkthroughs.md`, and the README once individual milestones land so readers can trace how Growth-Mimetic functionality manifests in user-facing workflows.

By layering growth-aware mechanics on top of PERSIST’s persistence stack, these technologies deliver agents that grow, feel, and co-evolve with the richness of living ecosystems.
