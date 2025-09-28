# PERSIST
A ml/ai framework for persistence. 
Here’s a minimal-but-complete framework you can implement. It fuses (1) explicit homeostasis, (2) a viability kernel/shield, and (3) intrinsic persistence (empowerment or surprise minimization) on top of ordinary task reward.
PERSIST: A Persistence-Centric Agent Framework
1) Core concepts
State s_t = [o_t, x_t]: external observation o_t + internal/homeostatic variables x_t (e.g., energy, temperature, integrity).


Dynamics x_{t+1} = f_\theta(x_t, a_t, o_t): learned or known internal-state transition.


Viability set \mathcal{V} = \{x: g_i(x)\le0,\ i=1..m\}: constraints that define “alive.”


Persistence objective

 \max_\pi\ \mathbb{E}\sum_t \gamma^t\big[\,R_{\text{task}}(s_t,a_t)\ +\ \lambda_H R_{\text{homeo}}(x_t)\ +\ \lambda_I R_{\text{intr}}(s_t)\,\big]

 subject to x_t \in \mathcal{V} \forall t (enforced by a shield).


Homeostasis term R_{\text{homeo}}(x)=-\sum_j w_j\,(x_j-\mu_j)^2.


Intrinsic term options


Empowerment (channel capacity proxy): R_{\text{intr}} \approx \widehat{I}(A_{t:t+k};S_{t+k}|S_t).


Surprise minimization: R_{\text{intr}} = -\text{NLL}{\phi}(o{t+1}|s_t,a_t) under a world model.


2) Modules
World Model p_\phi(o_{t+1}|s_t,a_t) (+ latent if needed).


Internal Model f_\theta: predicts x_{t+1} and constraint margins g(x).


Viability Approximator \hat{\mathcal{V}}_\psi: classifier/regressor that outputs margin m(x) (positive = safe).


Safety Shield \mathcal{S}: action filter projecting a_t to nearest a’t s.t. expected x{t+1}\in\hat{\mathcal{V}}_\psi with confidence \ge \alpha.


Policy/Value \pi_\omega(a|s),\ V_\omega(s) (any actor-critic).


Intrinsic Estimator


Empowerment: maximize \log |\det \partial s_{t+k}/\partial a_{t:t+k}| (local Jacobian proxy) or InfoNCE k-step mutual information.


Surprise: negative predictive loss from world model.


Replay Buffers: task, model, and “near-boundary” buffer for constraint learning.


Curriculum: start wide tolerances; progressively tighten \mathcal{V} and increase \lambda_H,\lambda_I.


3) Training loop (high level)
for episode in range(E):
    s = env.reset()
    x = init_internal_state()
    for t in range(T):
        a = policy.sample(s)
        # Safety shield
        a_safe = shield.project(a, s, internal_model, viability_approx, conf=alpha)

        # Step env + update internal state
        o_next, r_task, done, info = env.step(a_safe)
        x_next = internal_model.predict_next(x, a_safe, o_next)  # differentiable

        # Intrinsic and homeostatic rewards
        r_homeo = -((x_next - mu)**2 * w).sum()
        r_intr = intrinsic.compute(s, a_safe, o_next)  # empowerment or surprise
        r_total = r_task + lambda_H * r_homeo + lambda_I * r_intr

        # Store
        replay.store(s, a_safe, r_total, o_next, x_next, done,
                     extras=dict(r_task=r_task, r_homeo=r_homeo, r_intr=r_intr))

        # Learn (periodically)
        if step % update_every == 0:
            world_model.update(replay_for_model)
            internal_model.update(near_boundary_buffer)
            viability_approx.update(near_boundary_buffer)
            policy.update(replay, worlds=(world_model, internal_model),
                          constraints=(viability_approx,))

        s, x = (np.concatenate([o_next, x_next]), x_next)
        if done: break
4) Safety shield (projection)
Predictive check: sample K rollouts from (f_\theta, p_\phi) for candidate a; estimate probability that x_{t+1:t+h}\in\hat{\mathcal{V}}_\psi.


If below \alpha, project: solve

 a’=\arg\min_{a’} \|a’-a\|\quad\text{s.t.}\quad \Pr(x_{t+1:t+h}\in\hat{\mathcal{V}}_\psi)\ge \alpha.

 Use CEM or gradient steps through differentiable models.


5) Intrinsic term—two ready paths
A) Empowerment (InfoNCE, short horizon)
Sample k-step action sequences A and resulting S’ from the world model.


Train a discriminator D_\eta(S’,A,S) with InfoNCE; use \log D_\eta as R_{\text{intr}}.


Backprop to policy via RL (or use model-based planning).


B) Surprise minimization (Active-inference-lite)
Use world-model NLL as surprise; add setpoint prior by penalizing predicted internal drift:

 R_{\text{intr}} = -\text{NLL}{\phi}(o{t+1}|s_t,a_t) - \beta \|x_{t+1}-\mu\|_2^2.


6) Viability learning
Label states using true constraints when available; otherwise, treat boundary hits (failures or alarms) as negative and far-from-boundary samples as positive; train \hat{\mathcal{V}}_\psi with focal loss.


Maintain a near-boundary buffer using high-margin uncertainty (e.g., 0 < m(x) < \epsilon) to constantly refine the frontier.


7) API sketch (PyTorch-ish)
class Homeostat:
    def __init__(self, mu, w): self.mu, self.w = mu, w
    def reward(self, x): return -((x - self.mu)**2 * self.w).sum(-1)

class ViabilityApprox(nn.Module):
    def forward(self, x):  # returns margin m(x)
        return self.net(x)

class Shield:
    def project(self, a, s, internal_model, viability, conf=0.95, horizon=H):
        if self.safe(a, s, internal_model, viability, conf, horizon): return a
        return self._cem_project(a, s, internal_model, viability, conf, horizon)

class Intrinsic:
    def __init__(self, mode="empowerment", **kwargs): ...
    def compute(self, s, a, o_next): ...

class PersistAgent:
    def step(self, s):
        a = self.policy.sample(s)
        a = self.shield.project(a, s, self.internal_model, self.viability)
        return a
8) Config (YAML)
env:
  name: "GridLife-v0"
  horizon: 512
internal_state:
  dims: ["energy","temp","integrity"]
  mu: [0.7, 0.5, 0.9]
  w:  [1.0, 0.5, 1.5]
viability:
  constraints:
    - "energy >= 0.2"
    - "temp in [0.3, 0.7]"
    - "integrity >= 0.6"
  shield:
    conf: 0.97
    horizon: 8
rewards:
  lambda_homeo: 0.7
  lambda_intr:  0.3
  intrinsic: "empowerment"  # or "surprise"
train:
  algo: "SAC"        # or PPO/TD3
  batch_size: 1024
  gamma: 0.995
  update_every: 64
  model_rollouts: 5
9) Minimal testbed (quick sanity environment)
2D grid with heat map and food tiles.


x = [energy,temp,integrity].


Energy decays; food raises energy; hot tiles raise temp; hazards reduce integrity.


Episode ends if any constraint violated; task reward = exploration bonus or item collection.


10) Metrics
Time-to-violation (survival steps).


Constraint satisfaction rate per variable.


Policy entropy under constraints (robustness).


Recovery rate: fraction of near-boundary states that return to safe core within h steps.


Empowerment/surprise curves vs. survival time.


11) Implementation order (fastest path)
Wrap an existing RL stack (SAC/PPO).


Add homeostat reward.


Train a simple viability classifier from failures; add shield with CEM projection over a learned f_\theta.


Plug in surprise (world-model NLL) first; then swap to empowerment InfoNCE if needed.


Add curriculum: tighten constraints + raise \lambda_H as policy stabilizes.


12) Extensions (when you need more)
Risk-sensitive RL: CVaR on violation probability.


Reach-avoid MPC: model-predictive layer with terminal set inside \mathcal{V}.


Adaptive setpoints: \mu learned via meta-RL for changing environments.


Partial observability: maintain belief over x with a filter (EKF/GRU).
Next steps
Implementing the PERSIST training process successfully requires a phased approach, starting with a standard reinforcement learning (RL) foundation and progressively adding the layers of persistence and safety.
Based on the provided document and recent advancements, here is a practical implementation plan, highlighting the key changes or modernizations you should consider.
A Phased Implementation Plan
This plan follows the logical order suggested by the document, from the simplest components to the most complex.
Phase 1: Foundational RL Setup
The goal here is to get a standard agent working in your environment before adding complexity.
 * Environment and State Definition:
   * Set up a minimal testbed environment as described in the document, such as a 2D grid with internal variables for energy, temperature, and integrity.
   * Define the full agent state s_t = [o_t, x_t], combining the external observation (o_t) with the internal/homeostatic variables (x_t).
 * Base RL Agent:
   * Implement a standard actor-critic agent like SAC or PPO. This will serve as your Policy/Value network (\pi_\omega).
   * At this stage, train the agent only on the task reward, R_{\text{task}}. The goal is to confirm that the basic RL loop is functioning correctly.
Phase 2: Integrating Persistence Rewards
Now, we introduce the core reward signals that encourage long-term survival.
 * Add the Homeostat:
   * Implement the homeostatic reward function: R_{\text{homeo}}(x) = -\sum_j w_j (x_j - \mu_j)^2. This is a simple module that calculates a penalty based on the squared distance of internal variables from their ideal setpoints, \mu.
   * Modify the total reward signal to be r_{total} = R_{\text{task}} + \lambda_H \cdot R_{\text{homeo}}. You will need to tune the lambda_homeo weight.
 * Add the Intrinsic Term:
   * You need a World Model (p_\phi) that predicts the next observation (o_{t+1}) given the current state and action.
   * Implement Surprise Minimization first, as it is the simplest path. The intrinsic reward is the negative log-likelihood (NLL) of the world model's prediction: R_{\text{intr}} = -\text{NLL}_\phi(o_{t+1}|s_t, a_t).
   * Update the total reward to include all three terms: r_{total} = R_{\text{task}} + \lambda_H R_{\text{homeo}} + \lambda_I R_{\text{intr}}.
Phase 3: Implementing the Viability Shield
This is the most critical and complex part, focused on enforcing hard safety constraints.
 * Train Support Models:
   * Internal Model (f_\theta): Train a model to predict the next internal state, x_{t+1}, given the current state and action. This is essential for the shield to look ahead.
   * Viability Approximator (\hat{\mathcal{V}}_\psi): Train a classifier that outputs a "margin of safety" for a given internal state x. To train this, use a dedicated "near-boundary" replay buffer that stores experiences where the agent nearly violated a constraint. This helps the model focus on the most critical decision boundaries.
 * Implement the Shield:
   * Create the Safety Shield (\mathcal{S}) module.
   * For every action a sampled by the policy, the shield performs a predictive check: it uses the internal model and viability approximator to predict if a will lead to an unsafe state (x_{t+1} \notin \hat{\mathcal{V}}_\psi).
   * If the action is predicted to be unsafe with a certain confidence, the shield must solve for a new action, a', that is both safe and as close as possible to the original a. This can be done with optimization methods like Cross-Entropy Method (CEM).
 * Introduce a Curriculum:
   * Do not start with strict constraints. Begin with wide tolerances for viability and low weights for the persistence rewards. As the policy improves and stabilizes, progressively tighten the constraints and increase \lambda_H and \lambda_I.
Recommended Changes for a Modern & Successful Implementation
The original framework is excellent but computationally heavy. Here are the key changes to make it more efficient and robust:
 * Change 1: Use a More Powerful Latent World Model.
   * Instead of a simple model that predicts the next raw observation, use a modern latent dynamics model like DreamerV3. These models learn a compressed, abstract representation of the world, which is far more sample-efficient and computationally tractable. This single upgrade improves the quality of the surprise-based intrinsic reward and provides a more robust foundation for the shield's predictions.
 * Change 2: Simplify Intrinsic Motivation with RND.
   * While Empowerment is a powerful idea, the InfoNCE method described can be complex and unstable to train. A more modern and lightweight approach is Random Network Distillation (RND). It provides a strong exploration signal by rewarding the agent for visiting novel states and is much simpler to implement than empowerment, reducing overall complexity.
 * Change 3: Amortize the Safety Shield's Computation.
   * The original plan's shield solves a costly optimization problem for every potentially unsafe action. A significant change would be to amortize this calculation. You can train a separate "safety network" that learns to perform the projection from an unsafe action a to a safe action a' directly. This converts a slow, iterative optimization into a fast, single forward pass through a network, drastically speeding up runtime.
 * Change 4: Learn Viability from Safe Demonstrations.
   * The framework's method for learning the viability boundary relies on observing failures or near-failures. A safer and often more effective approach is to use Imitation Learning from demonstrations. By showing the agent examples of an expert operating safely, the ViabilityApproximator can learn the safe operating region without the agent ever needing to risk a catastrophic failure.


***more to do: 
*You’ve got a solid spine. What’s missing are the pieces that turn “survive this episode” into “survive across regimes, failures, and time.” Gap list with concrete adds:

1) Hard-safety math (beyond a learned viability classifier)
	•	Control Barrier Functions (CBF) / CLF-CBF-QP layer for analytic safety guarantees on continuous controls.
Add: components/cbf_layer.py (QP solver wrapper), components/dynamics_adapter.py (linearization).
Config: safety.cbf:{enabled, relax_penalty, delta}.
	•	Hamilton–Jacobi Reachability for small systems to compute true viable sets (offline) and distill to \hat{\mathcal V}_\psi.
Add: tools/hj_reachability/ (offline precompute) + scripts/distill_viability.py.

2) Constraint enforcement with dual control
	•	Lagrangian/chance constraints (primal–dual) so the agent adapts \lambda_H automatically to hold a target violation rate.
Add: components/constraint_manager.py (dual ascent updating multipliers), integrates into loss.
Metrics: violations.moving_avg, dual.lambda_history.

3) Out-of-distribution + anomaly defense
	•	Runtime OOD gate on observations and internal x: energy-based scores / MC-Dropout / ensembles.
Action: fall back to shielded “safe policy” or MPC when OOD.
Add: components/ood_detector.py, policies/safe_fallback.py.
Config: ood:{method, threshold, fallback="mpc|rule"}.

4) Continual / lifelong learning without catastrophic forgetting
	•	Stability–plasticity: EWC / MAS / L2-SP + rehearsal buffer + periodic consolidation checkpoints.
Add: components/continual.py with penalties + buffers/rehearsal.py.
Schedule: consolidate every N episodes; freeze safety-critical heads.

5) Self-maintenance behaviors (not just avoiding failure)
	•	Maintenance actions & schedules (refuel, cool-down, self-repair) modeled as explicit subgoals with costs.
Add: env/maintenance_tasks.py, reward shaping for deferred maintenance avoidance.
Planner: encode periodic maintenance windows in MPC terminal costs.

6) Resource accounting / bounded rationality
	•	Compute/energy budgets as first-class state with penalties and termination.
Add: components/budget_meter.py (tracks FLOPs/latency/energy), hooks into homeostat.
Config: budgets:{compute_ms_per_step, energy_cap}.

7) Adversarial robustness & red-team loops
	•	Adversarial training for observation/action perturbations; worst-case rollout sampling.
Add: components/adversary.py (PGD/RS), trainers/robust_trainer.py.
Metric: worst-k CVaR survival under perturbations.

8) Interpretability for the safety path
	•	Causal/probing heads that predict each constraint margin g_i(x) from latent features; saliency on shield decisions.
Add: components/safety_probe.py + attribution reports in utils/reporting.py.
Artifact: “why shield blocked” JSON per step.

9) Formal persistence across restarts (system-level)
	•	Checkpointing & roll-forward policy with integrity checks; degraded-mode controller if load fails.
Add: systems/persistence.py (atomic save, hash, warm-start), policies/degraded_mode.py.
Config: persistence:{interval_steps, retain_n, crc32=true}.

10) Population-level persistence
	•	Ecological diversity & redundancy: population-based training (PBT), niching, ensemble safety votes.
Add: population/pbt.py, population/ensemble_shield.py.
Metric: fleet survival under correlated shocks.

11) Multi-agent viability (competition/cooperation)
	•	Shared-resource constraints and market/auction or control-barrier coupling between agents.
Add: multiagent/resource_allocator.py (VCG or proportional), multiagent/cbf_coupler.py.
Tests: tragedy-of-commons scenarios.

12) Verification & falsification harness
	•	Property-based testing for safety invariants; scenario fuzzing and falsification (CEM/SMT).
Add: tests/spec_safety_test.py, tools/fuzz_scenarios.py.
CI: block merges on invariant breaches.

13) Evaluation under regime shift
	•	Shift suite: weathered dynamics, sensor dropouts, parameter drifts, rare hazards.
Add: benchmarks/persist_suite/ with YAML scenarios.
Metrics: survival @ shift, recovery time, safe-return rate after boundary hit.

14) Data & config contract for constraints
	•	Schema to declare variables, units, bounds, recovery dynamics, maintenance ops.
Add: schemas/viability.schema.json; validator in utils/validate_config.py.
Enforce: refuse run if constraints underspecified.

15) Governance hooks (practical ops)
	•	Runtime monitors (Prometheus/OpenTelemetry), alerting on leading indicators (near-boundary density ↑).
Add: ops/telemetry.py, ops/alerts.yml.
Dashboard: survival curves, dual multipliers, shield trigger rate.

⸻

Concrete patch list (minimal to cover the gaps)
	•	components/cbf_layer.py, components/constraint_manager.py, components/ood_detector.py
	•	components/continual.py, buffers/rehearsal.py
	•	systems/persistence.py, policies/degraded_mode.py
	•	population/ensemble_shield.py
	•	benchmarks/persist_suite/ + tests/spec_safety_test.py
	•	schemas/viability.schema.json + utils/validate_config.py

Config deltas

safety:
  cbf: {enabled: true, relax_penalty: 10.0}
  chance_constraint: {target_violation_rate: 0.005, dual_lr: 5e-4}
ood:
  method: "energy"   # energy|mc_dropout|ensemble
  threshold: -5.0
continual:
  ewc_lambda: 5.0
  consolidate_every: 5000
budgets:
  compute_ms_per_step: 10
  energy_cap: 1.0
persistence:
  checkpoint_every: 2000
  retain: 5
population:
  ensemble_size: 3
  vote: "veto_if_any_unsafe"
eval:
  shift_suite: ["sensor_dropout", "dynamics_drift", "rare_hazard"]

Why these matter
	•	CBF/HJ gives safety guarantees you can’t get from a learned classifier alone.
	•	Dual control & chance constraints keep violations at a target rate without hand-tuning \lambda.
	•	OOD, continual, adversarial, maintenance = persistence under reality, not just the training distribution.
	•	System persistence + population redundancy handles crashes and correlated failures—the real killers of long-run survival.
	•	Spec tests + shift suite make “persistence” measurable and regressions obvious.

---ON 09.27.2025.@15.56.00----

What’s still missing

Here are the remaining gaps that separate episodic survival from true persistence across environments, failures, and time:

1. Formal verification hooks

You have property-based tests, but not formal reachability guarantees for the shield + internal model.

Add: components/cbf_layer.py + tools/hj_reachability/ so you can actually compute or bound safe sets analytically for small subsystems.

2. Constraint governance

Right now, constraint definitions live in YAML. They need first-class schema validation + runtime contracts.

Add: schemas/viability.schema.json + utils/validate_config.py.

Enforce: refuse training if constraints are underspecified (no units, no bounds).

3. Self-maintenance behaviors

Avoiding failure isn’t the same as planning proactive repairs.

Add explicit maintenance tasks (cool-down, refuel, self-repair) in the environment, with costs and scheduling.

This pushes the agent to learn deferred maintenance tradeoffs, critical for persistence.

4. Adversarial robustness

No red-team loop yet. Without it, persistence breaks under adversarial perturbations.

Add: components/adversary.py, trainers/robust_trainer.py.

Metric: worst-k CVaR survival under perturbations.

5. Interpretability for safety path

Right now, you log violations, but not why the shield blocked an action.

Add: components/safety_probe.py (predicts each constraint margin from latent features) + attribution reports.

Output JSON traces: "shield_reason": "temp_out_of_bounds".

6. Multi-agent persistence

Your current system is single-agent. True persistence usually requires coexistence under shared resources.

Add: multiagent/resource_allocator.py, multiagent/cbf_coupler.py.

Test on tragedy-of-commons environments.

7. Governance + telemetry

You have metrics, but not ops-grade monitoring.

Add: ops/telemetry.py, ops/alerts.yml.

Hook into Prometheus/OpenTelemetry to trigger alerts when “near-boundary density” spikes.

Minimal patch list left to implement

components/cbf_layer.py + components/dynamics_adapter.py

tools/hj_reachability/ + scripts/distill_viability.py

schemas/viability.schema.json + utils/validate_config.py

env/maintenance_tasks.py

components/adversary.py + trainers/robust_trainer.py

components/safety_probe.py + utils/reporting.py

multiagent/resource_allocator.py + multiagent/cbf_coupler.py

ops/telemetry.py + ops/alerts.yml

Config deltas to support these
maintenance:
  tasks: ["refuel", "cool_down", "repair"]
  penalty_costs: {refuel: 0.1, repair: 0.3}
adversarial:
  enabled: true
  method: "pgd"
  epsilon: 0.05
safety_probe:
  enabled: true
  explain: "json"
multiagent:
  resources: {food: 10, shelter: 5}
  coupling: "cbf"
telemetry:
  prometheus: true
  alert_thresholds:
    near_boundary_density: 0.15

Bottom line

You’ve got the bones of persistence: homeostasis, shields, intrinsic motivation.
You’ve added muscle: MPC, CVaR, ensembles, continual learning, OOD detection.
What’s left is the infrastructure that turns this from “trainable agent” into a true persistence system:

formal safety proofs,

explicit maintenance,

adversarial hardening,

interpretability,

multi-agent coexistence,

ops governance.

That’s the difference between “survive the episode” and “survive in the wild.”

***
### Progress Updates
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
