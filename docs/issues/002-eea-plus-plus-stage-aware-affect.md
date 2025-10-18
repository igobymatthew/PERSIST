# Issue: Implement Emotional Equilibrium Atlas++ (Stage-Aware Affect)

## Summary
`docs/growth_mimetic_technologies.md` proposes extending the Emotional Equilibrium Architecture into EEA++ so affect buffers, replay, and telemetry become stage-aware. While the current `LifeStageManager` surfaces affect bounds and telemetry gauges (`ops/telemetry.py`), the EEA agents do not yet enforce or adapt to those bounds. This issue delivers the missing integration.

## Implementation Plan
1. **Stage-conditioned affect buffers**
   - Update `agents/eea_agent.py` to request the active stage’s affect targets from the `LifeStageManager` (accessible via the coordinator or dependency injection).
   - Modify affect ratio computations to clamp and regularize toward the stage-specific ranges.
   - Extend replay sampling so transitions include the active `life_stage` label.
2. **Recovery routines & fire events**
   - During `FireEvent` resets, seed the EEA buffers with context-weighted experiences that match the new stage.
   - Add fast-path recovery heuristics that prioritize experiences near the new stage’s target ratios.
3. **Telemetry and observability**
   - Emit stage-scoped affect metrics via `TelemetryManager.update_life_stage`, ensuring happiness/fear bands appear in logs and Prometheus gauges.
   - Add debug logging so developers can trace when the agent deviates from stage targets.
4. **Testing & docs**
   - Write unit tests in `tests/test_eea_agent.py` that simulate stage transitions and confirm affect ratios stay within configured bounds.
   - Update `docs/EEA.md` with an “EEA++” section that explains the new behavior and references relevant configuration fields.

## Acceptance Criteria
- EEA agents adapt affect ratios based on the active life stage without regressing existing behaviors.
- Replay buffers store and expose the active stage label for learning.
- Telemetry dashboards show stage-aware affect bands in Prometheus metrics.
- Tests covering stage-conditioned affect logic pass locally and in CI.
- Documentation reflects the EEA++ workflow and configuration.
