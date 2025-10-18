# Issue: Integrate DreamerV3-Style Latent World Model

## Summary
The modernization plan in `docs/theory.md` recommends upgrading the existing latent world model to a DreamerV3-style architecture so surprise rewards and shield rollouts leverage a richer learned dynamics model. Although we currently train a lightweight latent model (`components/latent_world_model.py`), it lacks recurrent state-space modeling, imagination rollouts, and value heads that Dreamer provides. This issue tracks the work required to land that upgrade.

## Status
✅ Completed in `components/dreamer_world_model.py` with configuration wiring (`world_model.type: "dreamer"`) and Dreamer-specific tests (`tests/test_dreamer_world_model.py`). Trainers, the safety shield, and the MPC planner now fall back gracefully when the Dreamer model is disabled.

## Implementation Plan
1. **Design the RSSM backbone**
   - Implement a recurrent state-space model (RSSM) module with stochastic and deterministic latent states (see DreamerV3).
   - Add encoder/decoder towers that mirror the observation space used in `environments/grid_life.py`.
   - Provide configuration knobs in `config.yaml` under a new `dreamer` section (latent sizes, KL scales, horizon length).
2. **Extend training utilities**
   - Update the trainer in `utils/trainer.py` (and `utils/robust_trainer.py`) to support Dreamer-style imagination rollouts when `rewards.intrinsic` is `"surprise"`.
   - Add optimizer scheduling hooks so the Dreamer model, policy, and value heads follow separate learning rates.
   - Ensure replay buffers expose sequences with the `sequence_length` needed for RSSM updates.
3. **Expose imagination to shield and planners**
   - Teach `components/shield.py` and the MPC planner to call the Dreamer model’s rollout API when available, falling back to the existing latent model otherwise.
   - Provide helper methods in `components/latent_world_model.py` (or a new `components/dreamer_world_model.py`) that export predicted viability margins for candidate action sequences.
4. **Validation & documentation**
   - Add unit tests in `tests/` that confirm RSSM forward passes, KL balancing, and imagination rollouts work with dummy data.
   - Document the new pipeline in `docs/theory.md` (implementation status) and add a usage walkthrough to `docs/use_case_walkthroughs.md`.

## Acceptance Criteria
- DreamerV3-style RSSM modules live under `components/` with configuration handled in `config.yaml`.
- Trainers can switch between the existing latent model and the Dreamer variant without breaking current tests.
- Shield and MPC planners consume imagination rollouts from the Dreamer model when enabled.
- New tests covering Dreamer components pass in CI.
- Documentation updates describe how to enable and tune the Dreamer world model.
