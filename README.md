# PERSIST
A ml/ai framework for persistence. 
Here’s a minimal-but-complete framework you can implement. It fuses (1) explicit homeostasis, (2) a viability kernel/shield, and (3) intrinsic persistence (empowerment or surprise minimization) on top of ordinary task reward.

## Getting Started

This project includes an interactive command-line interface (CLI) to simplify running experiments.

If you are designing training loops, see [`docs/training_resilience.md`](docs/training_resilience.md) for optimizer and LoRA adapter guidance tailored to PERSIST's fire-reset philosophy.

For concrete, step-by-step examples of running the framework, follow the new [hands-on walkthroughs](docs/use_case_walkthroughs.md).

### 1. Environment Setup

It is recommended to use Conda to manage dependencies.

1.  **Create the Conda environment** from the `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    ```
2.  **Activate the environment**:
    ```bash
    conda activate persist
    ```

### 2. Running an Experiment

Once the environment is activated, you can start the interactive CLI:

```bash
python main.py
```

The CLI will guide you through several choices to configure your experiment:
- **Standard single-agent experiment**: A basic setup, ideal for beginners.
- **Multi-agent experiment**: A setup for running simulations with multiple agents.
- **Robust agent experiment**: A setup that includes adversarial training and advanced safety features.
- **Custom experiment**: Allows you to pick and choose features for a custom configuration.
- **Run from `config.yaml`**: The original behavior, which runs the experiment directly from the `config.yaml` file without any modifications.

After you make your selections, the final configuration will be displayed for confirmation before the experiment begins.


### Progress Updates
*   **2025-09-29T10:35:25-04:00**: Captured the long-term roadmap and review guidance in dedicated documentation.
    *   Authored `docs/roadmap.md` to record wins, refactor priorities, testing plans, and milestone checklists for future contributors.
    *   Documented actionable steps for README modularization, API hygiene, CI coverage, and developer experience improvements.
    *   Highlighted near-term verification, collaboration, and benchmarking tasks to keep the repository aligned with the architecture vision.
*   **2025-09-29T10:34:54-04:00**: Streamlined the top-level README around quick start usage.
    *   Removed internal review checklists so the README now emphasizes onboarding instructions and the progress journal.
    *   Focused the "Getting Started" narrative on the Conda workflow and interactive CLI entry point for running experiments.
    *   Preserved the historical changelog while paving the way for deeper docs to live in `docs/`.
*   **2025-09-29T10:32:17-04:00**: Deepened the CLI walkthrough for configuring training runs.
    *   Added interactive flows in `main.py` that let users tune epochs, update cadence, learning rates, and shield amortization thresholds before launching training.
    *   Enabled optional prompts for resizing network capacity and curriculum schedules, keeping advanced knobs accessible without editing YAML by hand.
    *   Synced defaults in `config.yaml` with the new questionnaire so saved configurations match the guided selections.
*   **2025-09-28T10:02:12+00:00**: Added automated data collection around the viability frontier.
    *   Implemented a configurable `NearBoundaryBuffer` (`buffers/near_boundary.py`) that captures internal states with safety
        probabilities near the decision boundary and replays them during updates.
    *   Extended the trainer and coordinator so the `ViabilityApproximator` (and ensemble variants) receive extra supervision
        on these ambiguous samples, improving shield accuracy without inducing extra risk.
    *   Updated `config.yaml` and the viability schema to expose `viability.near_boundary_buffer` controls and documented the
        feature in `docs/theory.md`.
    *   Added targeted unit tests for the buffer to guarantee sampling and filtering behave as expected.
*   **2025-09-28T09:14:00+00:00**: Improved developer experience by simplifying setup and usage.
    *   Added an `environment.yml` file to allow for easy environment setup using Conda.
    *   Created a "Getting Started" section in `README.md` to document the environment setup process and the existing interactive CLI.
*   **2025-09-28T08:44:41+00:00**: Implemented an interactive command-line interface (CLI) in `main.py`.
    *   The new CLI guides users through selecting an experiment type (e.g., standard single-agent, multi-agent, robust agent) or building a custom configuration.
    *   This eliminates the need for manual `config.yaml` editing for most use cases, making the framework more user-friendly.
    *   The underlying `ComponentFactory` and validation logic were refactored to support dynamic configuration generation.
*   **2025-09-28T07:48:00+00:00**: Improved framework robustness and observability.
    *   Strengthened the testing framework by adding comprehensive test suites for the multi-agent environment (`tests/test_multiagent_env.py`) and the safety shield (`tests/test_shield.py`).
    *   Implemented a fail-fast configuration validation system by creating a master JSON schema (`schemas/config.schema.json`) and integrating it into the application's startup sequence.
    *   Added detailed logging to the `Shield` and `ResourceAllocator` components to provide debug traces for safety and resource management decisions.
*   **2025-09-28T05:07:24+00:00**: Completed major framework components and documentation alignment.
    *   Replaced the mock Hamilton-Jacobi reachability analysis with a functional, grid-based backward reachability implementation in `tools/hj_reachability/compute_viability.py`.
    *   Expanded the constraint governance feature by updating `schemas/viability.schema.json` to a more detailed, structured format and modified `config.yaml` and `environments/grid_life.py` to use the new schema.
    *   Refactored the self-maintenance logic into a dedicated `MaintenanceManager` class in `environments/maintenance_tasks.py` for improved modularity.
    *   Updated the `README.md` to accurately reflect the current state of the project, including removing outdated sections and updating configuration examples.
*   **2025-09-27T18:49:00+00:00**: Implemented the "Governance + telemetry" feature.
    *   Created an `ops/` directory for operational components.
    *   Added a `TelemetryManager` (`ops/telemetry.py`) that uses `prometheus-client` to expose key training metrics (e.g., survival time, constraint violations, shield trigger rate) via an HTTP endpoint.
    *   Defined a set of sample alerting rules in `ops/alerts.yml` for use with a Prometheus server.
    *   Integrated the `TelemetryManager` into the main training loop (`systems/coordinator.py` and `utils/trainer.py`) to provide real-time monitoring of the agent's performance and stability.
    *   Added a `telemetry` section to `config.yaml` to enable and configure the feature.
*   **2025-09-27T15:19:51+00:00**: Implemented a multi-agent persistence architecture (CTDE).
    *   Created `MultiAgentGridLifeEnv` with a dictionary-based API for managing multiple agents in a shared world.
    *   Implemented a `SharedSAC` agent using parameter sharing and role embeddings for efficient homogeneous multi-agent learning.
    *   Added a `MultiAgentTrainer` and `ReplayMA` buffer to handle the centralized training loop.
    *   Integrated a `ResourceAllocator` for proportional resource distribution and a `CBFCoupler` for collision avoidance.
    *   The entire multi-agent system is configurable via the `multiagent` and `agent_types` sections in `config.yaml`.
*   **2025-09-27T13:32:29+00:00**: Implemented interpretability for the safety path.
    *   Added a `SafetyProbe` component (`components/safety_probe.py`) to predict individual constraint margins, providing insight into the agent's safety status.
    *   Created a `SafetyReporter` (`utils/reporting.py`) to generate detailed JSON logs explaining why the safety shield modified an action.
    *   The environment now calculates and provides true constraint margins, which are used to train the probe.
    *   The feature is fully configurable via the `safety_probe` section in `config.yaml`.
*   **2025-09-27T13:29:35+00:00**: Implemented adversarial robustness.
    *   Added an `Adversary` component (`components/adversary.py`) that uses PGD to generate adversarial examples.
    *   Created a `RobustTrainer` (`utils/robust_trainer.py`) to handle adversarial training.
    *   The `SAC` agent's critic is now trained on these adversarial examples to improve robustness.
    *   The feature is configurable via the `adversarial` section in `config.yaml`.
*   **2025-09-27T13:13:49+00:00**: Implemented self-maintenance behaviors.
    *   Added "refuel," "cool-down," and "repair" stations to the `GridLifeEnv`.
    *   These tasks incur a small configurable penalty, encouraging the agent to learn strategic resource management.
    *   Updated `config.yaml` with a `maintenance` section to control these features.
*   **2025-09-27T13:07:11+00:00**: Implemented the Hamilton-Jacobi (HJ) Reachability analysis scaffolding.
    *   Created `tools/hj_reachability/compute_viability.py` as a placeholder for offline viable set computation.
    *   Added `scripts/distill_viability.py` to train the `ViabilityApproximator` by distilling knowledge from the pre-computed set.
    *   This establishes the workflow for using formal methods to define safety and then transferring that knowledge to a neural network.
*   **2025-09-27T12:40:00+00:00**: Implemented the "Resource Accounting / Bounded Rationality" feature.
    *   Created a `BudgetMeter` component (`components/budget_meter.py`) to track resource consumption (e.g., energy, computation time) over an episode.
    *   Integrated the `BudgetMeter` into the main training loop (`utils/trainer.py`), which now terminates an episode and applies a configurable penalty if the agent's budget is exhausted.
    *   Added a `budgets` section to `config.yaml` to allow enabling and configuring this feature.
*   **2025-09-27T12:25:30+00:00**: Implemented several key system-level features to enhance robustness and verifiability.
    *   Added a `PersistenceManager` (`systems/persistence.py`) for atomic checkpointing and a `DegradedModePolicy` (`policies/degraded_mode.py`) for safe fallbacks.
    *   Established a verification harness with property-based safety tests (`tests/spec_safety_test.py`) and a scenario fuzzer (`tools/fuzz_scenarios.py`).
    *   Created an evaluation suite (`benchmarks/persist_suite/`) with scenarios for regime shifts like sensor dropout, dynamics drift, and rare hazards.
    *   Implemented a data contract (`schemas/viability.schema.json`) and a validation script (`utils/validate_config.py`) to ensure configuration integrity.
*   **2025-09-27T19:16:45+00:00**: Implemented the "Population-level persistence" extension by adding an `EnsembleShield`.
    *   Created a `population/` directory to house all population-based components.
    *   Implemented the `EnsembleShield` (`population/ensemble_shield.py`), which aggregates safety votes from multiple `ViabilityApproximator` models.
    *   The `EnsembleShield` supports multiple voting mechanisms, such as "veto_if_any_unsafe" and "majority_vote."
    *   Updated the `ComponentFactory` and `config.yaml` to allow for the creation and configuration of the `EnsembleShield`.
    *   Modified the `Trainer` to train all models in the `viability_ensemble`.
*   **2025-09-27T12:07:30+00:00**: Implemented a continual learning system to mitigate catastrophic forgetting.
    *   Created a `RehearsalBuffer` (`buffers/rehearsal.py`) to store past experiences using reservoir sampling.
    *   Implemented a `ContinualLearningManager` (`components/continual.py`) that uses Elastic Weight Consolidation (EWC) to protect important neural network weights.
    *   Integrated the EWC penalty into the SAC agent's loss function and added a periodic consolidation step to the main training loop.
*   **2025-09-27T18:00:00+00:00**: Implemented advanced safety and robustness features as outlined in the roadmap.
    *   Added a Control Barrier Function (CBF) layer (`components/cbf_layer.py`) to provide formal safety guarantees by solving a Quadratic Program to filter actions. This includes a `DynamicsAdapter` (`components/dynamics_adapter.py`) for linearizing the internal model.
    *   Implemented a `ConstraintManager` (`components/constraint_manager.py`) that uses dual ascent to automatically tune homeostatic penalty multipliers, enforcing a target violation rate without manual parameter tuning.
    *   Introduced an energy-based Out-of-Distribution (OOD) detector (`components/ood_detector.py`) to identify novel states and a `SafeFallbackPolicy` (`policies/safe_fallback.py`) to ensure the agent takes a safe action when encountering them.
*   **2025-09-27T17:31:45+00:00**: Refactored the codebase for improved modularity, maintainability, and performance.
    *   Introduced a `ComponentFactory` (`utils/factory.py`) to centralize the initialization of all framework components.
    *   Created a `Trainer` class (`utils/trainer.py`) to encapsulate the main training loop and update logic, simplifying `train.py`.
    *   Added device management for GPU acceleration, automatically moving models and tensors to a CUDA device if available.
    *   Optimized the data pipeline by modifying the `ReplayBuffer` to return tensors directly on the correct device, reducing overhead.
*   **2025-09-27T16:44:07+00:00**: Established a testing framework and added unit tests for core components.
    *   Created a `tests/` directory to house all test suites.
    *   Added `tests/test_components.py` with unit tests for `Homeostat`, `ViabilityApproximator`, and `InternalModel`.
    *   These tests verify the correctness of reward calculation, model training, and prediction logic, improving the project's robustness and maintainability.
*   **2025-09-27T08:11:30+00:00**: Implemented a comprehensive evaluation and metrics logging system.
    *   Created an `Evaluator` class in `utils/evaluation.py` to handle logging.
    *   The system now tracks and logs key metrics from the `README.md` file, including survival time, constraint satisfaction rates, and policy entropy.
    *   Metrics are saved to `training.log` in a structured JSON format for analysis.
    *   The main training loop in `train.py` was updated to integrate the evaluator.
*   **2025-09-27T07:22:55+00:00**: Implemented the "Reach-avoid MPC" extension.
    *   Created a `ReachAvoidMPC` component (`components/reach_avoid_mpc.py`) that uses model-predictive control to plan safe, goal-oriented actions.
    *   The MPC planner uses the `LatentWorldModel`, `InternalModel`, and `ViabilityApproximator` to simulate future trajectories and select the best action sequence via the Cross-Entropy Method (CEM).
    *   Created a new `MPCAgent` (`agents/mpc_agent.py`) to integrate the planner into the training loop.
    *   Added an `mpc` section to `config.yaml` to enable and configure the MPC agent.
*   **2025-09-27T14:08:18+00:00**: Implemented the "Risk-sensitive RL" extension.
    *   Created a `CVAR_SAC` agent (`agents/cvar_sac.py`) that optimizes the Conditional Value at Risk (CVaR) of the return distribution, making the agent risk-averse.
    *   Implemented a `DistributionalCritic` that learns the return distribution using quantile regression.
    *   The actor's objective is to maximize the CVaR of the critic's predicted distribution.
    *   The feature is configurable in `config.yaml` via the `risk_sensitive` section, allowing control over risk-aversion.
*   **2025-09-27T13:59:05+00:00**: Implemented the "Adaptive setpoints" extension.
    *   Created a `MetaLearner` component (`components/meta_learner.py`) to dynamically adjust homeostatic setpoints (`mu`) based on long-term performance.
    *   Integrated the `MetaLearner` into the main training loop (`train.py`), allowing the agent to adapt its internal targets.
    *   Added a `meta_learning` section to `config.yaml` to control this feature.
*   **2025-09-27T06:09:46+00:00**: Implemented the 'Empowerment' intrinsic reward module as described in the framework's core concepts.
    *   Created a new `Empowerment` component (`components/empowerment.py`) that uses a contrastive discriminator (InfoNCE) to calculate intrinsic rewards.
    *   The component is trained on-the-fly using rollouts from the `LatentWorldModel`.
    *   Updated `train.py` to support 'empowerment' as a configurable intrinsic reward method.
    *   Added an `empowerment` section to `config.yaml` for hyperparameters.
*   **2025-09-26 12:23:24.130399**: Completed Phase 2 of the implementation plan by integrating a surprise-based intrinsic reward.
    *   Created a `WorldModel` component (`components/world_model.py`) using PyTorch to predict the next observation.
    *   The prediction error (MSE) of the `WorldModel` is used as the surprise reward (`r_intr`).
    *   The main training loop (`train.py`) was updated to initialize and train the `WorldModel`.
    *   The total reward was updated to `r_total = r_task + lambda_H * r_homeo + lambda_I * r_intr`.
    *   Added `torch` and `pyyaml` to `requirements.txt`.
*   **2025-09-26 14:54:16.787780**: Completed Phase 3 of the implementation plan by integrating the Viability Shield.
    *   Created an `InternalModel` (`components/internal_model.py`) to predict the next internal state.
    *   Created a `ViabilityApproximator` (`components/viability_approximator.py`) to predict the safety of a state.
    *   Created a `Shield` (`components/shield.py`) to filter unsafe actions.
    *   Updated the `ReplayBuffer` to store internal states and viability labels.
    *   Integrated all new components into the main training loop (`train.py`), which now uses the shield to select safe actions and trains the new models.
*   **2025-09-26 15:17:41.792207**: Implemented "Change 2" from the modernization plan by replacing the surprise-based intrinsic reward with Random Network Distillation (RND).
    *   Created an `RND` component (`components/rnd.py`) that calculates intrinsic reward based on the prediction error of a randomly initialized target network.
    *   Updated the main training loop (`train.py`) to use the `RND` component for intrinsic motivation.
    *   Modified `config.yaml` to allow selecting `"rnd"` as the intrinsic reward type.
*   **2025-09-26 18:18:45.834302**: Implemented "Change 3" from the modernization plan by amortizing the Safety Shield's computation.
    *   Created a `SafetyNetwork` component (`components/safety_network.py`) that learns to project unsafe actions to safe ones.
    *   Updated the `ReplayBuffer` to store both unsafe and safe actions to create a training dataset for the `SafetyNetwork`.
    *   Modified the `Shield` to operate in two modes: a `search` mode for data collection and an `amortized` mode that uses the trained `SafetyNetwork` for fast projection.
    *   Updated the main training loop (`train.py`) to train the `SafetyNetwork` and switch the shield's mode after a configured number of steps.
    *   Added a `safety_network` section to `config.yaml` to control the new component's hyperparameters.
*   **2025-09-26 18:52:38.574183**: Implemented "Change 1" from the modernization plan by replacing the simple world model with a latent dynamics model.
    *   Created a `LatentWorldModel` (`components/latent_world_model.py`) with separate `Encoder`, `TransitionModel`, and `Decoder` modules.
    *   The model is trained on reconstruction and dynamics prediction losses.
    *   The intrinsic reward is now calculated as the reconstruction error from the predicted latent state, providing a more robust surprise signal.
    *   Updated `train.py` to use the new latent model and switched `config.yaml` to use `"surprise"` reward.
*   **2025-09-26 19:15:19.784624**: Implemented "Change 4" from the modernization plan by enabling the `ViabilityApproximator` to learn from safe demonstrations.
    *   Created a `DemonstrationBuffer` (`components/demonstration_buffer.py`) to load and sample expert data.
    *   Modified the `ViabilityApproximator` to include a `train_on_demonstrations` method, allowing it to pre-train on safe states.
    *   Updated the main training loop (`train.py`) to use the new buffer and training method.
    *   Added a `demonstrations` section to `config.yaml` to specify the data path.
*   **2025-09-26 20:20:23.540386**: Implemented curriculum learning as described in the "Implementation order."
    *   Added a `CurriculumScheduler` to `train.py` to gradually increase the weights of persistence rewards (`lambda_homeo`, `lambda_intr`) and tighten the environment's viability constraints over a configurable number of steps.
    *   The `GridLifeEnv` was updated to support dynamic constraint changes.
    *   Added a `curriculum` section to `config.yaml` to control the new scheduling feature.
*   **2025-09-26 20:49:35.720744**: Implemented the "Partial Observability" extension from the roadmap.
    *   Created a `StateEstimator` component (`components/state_estimator.py`) with a GRU network to predict the internal state from external observations.
    *   Modified the `GridLifeEnv` to optionally hide the true internal state, simulating partial observability.
    *   Updated `train.py` to use the `StateEstimator` when partial observability is enabled, feeding the estimated state to the agent and other components.
    *   Enhanced the `ReplayBuffer` to support sampling of contiguous sequences required for training the recurrent estimator.
    *   Added `state_estimator` and `partial_observability` configurations to `config.yaml`.
