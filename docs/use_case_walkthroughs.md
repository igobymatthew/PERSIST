# PERSIST Hands-On Walkthroughs

These walkthroughs provide concrete, end-to-end exercises that demonstrate how to run PERSIST in common scenarios. Each guide ties together the interactive CLI, configuration surfaces, and observability tools so newcomers can see how the framework behaves in practice.

## Walkthrough 1: Launch a Baseline Survival Agent

**Goal:** Run the standard single-agent `GridLife` experiment and inspect persistence metrics.

1. **Prepare the environment**
   ```bash
   conda activate persist
   python -m pip install -e .
   ```
   Installing PERSIST in editable mode keeps the CLI and config schema in sync while you iterate.

2. **Start the guided CLI**
   ```bash
   python main.py
   ```
   - Choose **“Standard single-agent experiment”** when prompted.
   - Accept the defaults for replay buffer, shield, and telemetry when asked.

3. **Tune starter hyperparameters**
   - When prompted for **“Number of training epochs”**, enter `150` to keep the initial run short.
   - Leave the default learning rate and batch size; these values are coordinated with `config.yaml` and the schemas in `schemas/config.schema.json`.

4. **Review the generated summary**
   The CLI renders a bordered summary card showing:
   - `environment: GridLifeEnv`
   - `agent: SAC`
   - `safety: ViabilityApproximator + Shield`
   Confirm to start training.

5. **Monitor training output**
   - Follow the console log messages from `systems/coordinator.py` (e.g., `--- Starting Training Loop ---`, shield mode switches) to track episode progress and safety interventions.
   - When the `TelemetryManager` is enabled, the CLI prints the local Prometheus URL (default `http://localhost:8000`). Visit it in a browser to confirm metrics such as `persist_survival_steps` and `persist_shield_trigger_rate` are being exported.

6. **Capture results**
   - Training checkpoints are stored under `logs/checkpoints/` by the `PersistenceManager` (the directory is created on first run).
   - Structured logs are written to `training.log` in your working directory. Open the file to review reward trends and shield events.

## Walkthrough 2: Deploy a Multi-Agent Resilience Scenario

**Goal:** Run the centralized-training, decentralized-execution (CTDE) setup to manage a small fleet of agents.

1. **Select the experiment type**
   ```bash
   python main.py
   ```
   Choose **“Multi-agent experiment”**.

2. **Configure cooperative roles**
   - When prompted, accept the default `MultiAgentGridLifeEnv`.
   - For *number of agents*, enter `3` to spawn a triad (e.g., scout, engineer, firefighter).
   - Enable the **Shared SAC** policy when offered; this activates the `SharedSAC` implementation inside `agents/shared_sac.py`.

3. **Enable collective safety tooling**
   - Toggle on the **EnsembleShield** to aggregate safety votes, reinforcing the persistence guarantees across the team.
   - Leave adversarial training off for the first pass to shorten runtime.

4. **Inspect the multi-agent summary**
   Confirm that the CLI summary lists:
   - `agent_types: ['SharedSAC']`
   - `multiagent.roles: 3`
   - `population.shield: EnsembleShield`

5. **Run and observe coordination metrics**
   - During training, `ops/telemetry.py` exposes multi-agent metrics such as `persist_survival_steps`, `persist_shield_trigger_rate`, and `persist_steps_per_second`.
   - Episode summaries printed by `log_episode_data` report reward totals and violation flags so you can verify all agents remain within constraints.

6. **Validate cooperative behavior**
   - Run the targeted test suite to ensure the shared environment logic behaves as expected:
     ```bash
     python -m pytest tests/test_multiagent_env.py -q
     ```
   - Inspect the generated `training.log` entries for coordinated maintenance hand-offs and shield activations once the training episodes complete.

## Walkthrough 3: Stress-Test with Adversarial Disturbances

**Goal:** Combine adversarial training with the safety probe to understand how PERSIST responds to shocks.

1. **Clone a configuration preset**
   ```bash
   cp config.yaml config.adversarial.yaml
   ```
   This gives you a dedicated file to tweak without losing defaults.

2. **Edit the preset**
   Update the following sections:
   - `adversarial.enabled: true`
   - `adversarial.num_iter: 10` (more PGD steps for stronger perturbations)
   - `adversarial.epsilon: 0.05` (increase perturbation size cautiously)
   - `safety_probe.enabled: true`
   - `telemetry.port: 8001` (optional: move telemetry to a dedicated port if 8000 is in use)

3. **Run the custom config**
   ```bash
   python main.py
   ```
   Choose **“Run from config.yaml”**, then enter the path `config.adversarial.yaml` when prompted.

4. **Track shield justifications**
   - The `SafetyReporter` streams JSON lines to `safety_reports.log`. Each entry includes the predicted constraint margin and the fallback action chosen by `policies/safe_fallback.py`.
   - Console logs highlight significant state changes (e.g., shield mode switching to AMORTIZED, fire events triggering) so you can correlate them with spikes in the report.

5. **Compare adversarial vs. nominal performance**
   - Run the robustness-focused tests to confirm the adversarial trainer remains stable:
     ```bash
     python -m pytest tests/test_robust_trainer.py -q
     ```
   - Contrast key metrics in `training.log` against your baseline run (e.g., shield trigger rate, recovery penalties) to understand the impact of adversarial perturbations.

---

These walkthroughs provide concrete paths from `python main.py` to actionable artifacts—checkpoints, telemetry, and reports—so beginners can build intuition for how PERSIST enforces persistence across increasingly demanding scenarios.
