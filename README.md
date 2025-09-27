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

***
### Progress Updates
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
