import yaml
import numpy as np
import torch

from environments.grid_life import GridLifeEnv
from agents.persist_agent import PersistAgent
from agents.mpc_agent import MPCAgent
from components.homeostat import Homeostat
from components.replay_buffer import ReplayBuffer
from components.latent_world_model import LatentWorldModel
from components.internal_model import InternalModel
from components.viability_approximator import ViabilityApproximator
from components.rnd import RND
from components.empowerment import Empowerment
from components.shield import Shield
from components.safety_network import SafetyNetwork
from components.demonstration_buffer import DemonstrationBuffer
from components.state_estimator import StateEstimator
from components.meta_learner import MetaLearner

class CurriculumScheduler:
    def __init__(self, config):
        self.config = config.get('curriculum', {})
        if not self.config.get('enabled', False):
            self.enabled = False
            return

        self.enabled = True
        self.total_steps = self.config['steps']
        self.lambdas = {}
        for key in ['lambda_homeo', 'lambda_intr']:
            if key in self.config:
                self.lambdas[key] = (self.config[key]['start'], self.config[key]['end'])

        self.constraints = {}
        for const_conf in self.config.get('viability_constraints', []):
            self.constraints[const_conf['dim_name']] = (const_conf['start'], const_conf['end'])

    def _interpolate(self, start, end, progress):
        return start + (end - start) * progress

    def get_current_values(self, step):
        if not self.enabled:
            # This should not be called if not enabled, but as a safeguard:
            return {'lambda_homeo': 0, 'lambda_intr': 0, 'constraints': {}}

        progress = min(1.0, step / self.total_steps)

        current_values = {}
        # Interpolate lambdas
        for key, (start, end) in self.lambdas.items():
            current_values[key] = self._interpolate(start, end, progress)

        # Interpolate constraints
        current_constraints = {}
        for key, (start, end) in self.constraints.items():
            current_constraints[key] = self._interpolate(start, end, progress)

        current_values['constraints'] = current_constraints

        return current_values

def main():
    print("--- Starting Training Script ---")
    # Load configuration
    print("Loading configuration...")
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    print("✅ Configuration loaded.")

    # Initialize components
    print("\nInitializing environment...")
    env = GridLifeEnv(config)
    internal_dim = env.internal_dim
    external_obs_dim = env.external_obs_dim
    is_partially_observable = config.get('env', {}).get('partial_observability', False)
    print(f"✅ Environment initialized. Partial Observability: {is_partially_observable}")

    # Initialize State Estimator if in partial observability mode
    state_estimator = None
    if is_partially_observable:
        print("\nInitializing state estimator...")
        state_estimator = StateEstimator(
            obs_dim=external_obs_dim,
            act_dim=env.action_dim,
            internal_dim=internal_dim
        )
        print("✅ State estimator initialized.")

    print("\nInitializing homeostat...")
    # Initialize MetaLearner if enabled
    meta_learner_config = config.get('meta_learning', {})
    meta_learner = None
    initial_mu = config['internal_state']['mu']
    if meta_learner_config.get('enabled', False):
        print("\nInitializing MetaLearner...")
        meta_learner = MetaLearner(
            initial_mu=initial_mu,
            learning_rate=meta_learner_config.get('learning_rate', 0.01),
            update_frequency=meta_learner_config.get('update_frequency', 100)
        )
        print("✅ MetaLearner initialized.")
        initial_mu = meta_learner.get_setpoints()

    homeostat = Homeostat(
        mu=initial_mu,
        w=config['internal_state']['w']
    )
    print("✅ Homeostat initialized.")

    print("\nInitializing latent world model...")
    world_model = LatentWorldModel(
        obs_dim=env.observation_space_dim,
        act_dim=env.action_dim
    )
    print("✅ Latent world model initialized.")

    # Initialize Intrinsic Reward module based on config
    intrinsic_reward_module = None
    intrinsic_method = config['rewards']['intrinsic']
    if intrinsic_method == 'rnd':
        print("\nInitializing RND module...")
        intrinsic_reward_module = RND(obs_dim=env.observation_space_dim)
        print("✅ RND module initialized.")
    elif intrinsic_method == 'empowerment':
        print("\nInitializing Empowerment module...")
        empowerment_config = config.get('empowerment', {})
        intrinsic_reward_module = Empowerment(
            state_dim=world_model.latent_dim,
            action_dim=env.action_dim,
            k=empowerment_config.get('k', 4),
            hidden_dim=empowerment_config.get('hidden_dim', 256),
            lr=empowerment_config.get('lr', 1e-4)
        )
        print("✅ Empowerment module initialized.")
    elif intrinsic_method == 'surprise':
        print("\nUsing world model surprise as intrinsic reward.")
    else:
        print(f"⚠️ Unknown intrinsic reward method: {intrinsic_method}. Defaulting to 'surprise'.")
        config['rewards']['intrinsic'] = 'surprise'


    print("\nInitializing internal model...")
    internal_model = InternalModel(
        internal_dim=internal_dim,
        act_dim=env.action_dim
    )
    print("✅ Internal model initialized.")

    print("\nInitializing viability approximator...")
    viability_approximator = ViabilityApproximator(internal_dim=internal_dim)
    print("✅ Viability approximator initialized.")

    # --- Agent Initialization ---
    # The agent is initialized here, after all its potential dependencies (models) are ready.
    print("\nInitializing agent...")
    mpc_config = config.get('mpc', {})
    if mpc_config.get('enabled', False):
        agent = MPCAgent(
            latent_world_model=world_model,
            internal_model=internal_model,
            viability_approximator=viability_approximator,
            action_space=env.action_space,
            mpc_config=mpc_config
        )
    else:
        agent_obs_dim = external_obs_dim + internal_dim
        agent = PersistAgent(
            obs_dim=agent_obs_dim,
            act_dim=env.action_dim,
            act_limit=env.act_limit,
            risk_sensitive_config=config.get('risk_sensitive')
        )
    print("✅ Agent initialized.")


    # Initialize Safety Network
    print("\nInitializing safety network...")
    safety_network_config = config.get('safety_network', {})
    safety_network = SafetyNetwork(
        internal_dim=internal_dim,
        action_dim=env.action_dim,
        hidden_dim=safety_network_config.get('hidden_dim', 128)
    )
    print("✅ Safety network initialized.")

    print("\nInitializing safety shield...")
    shield = Shield(
        internal_model=internal_model,
        viability_approximator=viability_approximator,
        action_space=env.action_space,
        conf=config['viability']['shield']['conf'],
        safety_network=safety_network,
        mode='search'  # Start in 'search' mode to collect data
    )
    print("✅ Safety shield initialized.")

    print("\nInitializing replay buffer...")
    replay_buffer = ReplayBuffer(
        capacity=10000,
        obs_dim=env.observation_space_dim,
        action_dim=env.action_dim,
        internal_dim=internal_dim
    )
    print("✅ Replay buffer initialized.")

    # Initialize Demonstration Buffer if specified in config
    demo_buffer = None
    if 'demonstrations' in config and config['demonstrations'].get('filepath'):
        print("\nInitializing demonstration buffer...")
        demo_buffer = DemonstrationBuffer(
            filepath=config['demonstrations']['filepath']
        )
        if len(demo_buffer) == 0:
            print("⚠️  Demonstration buffer is empty. Check filepath in config.")
        else:
            print("✅ Demonstration buffer initialized.")

    print("\nInitializing curriculum scheduler...")
    scheduler = CurriculumScheduler(config)
    if scheduler.enabled:
        print("✅ Curriculum enabled.")
    else:
        print("ℹ️ Curriculum disabled, using fixed parameters.")

    print("\n--- ✅ All Components Initialized ---")

    # Get initial lambda values, which will be updated by the curriculum if enabled
    lambda_homeo = config['rewards']['lambda_homeo']
    lambda_intr = config['rewards']['lambda_intr']
    batch_size = config['train']['batch_size']
    update_every = config['train']['update_every']
    amortize_after_steps = config.get('train', {}).get('amortize_after_steps', 20000)
    total_steps = 0
    sequence_length = 16 # For training the state estimator

    num_episodes = 500
    for episode in range(num_episodes):
        # Reset environment and estimator state
        external_obs = env.reset()

        if not is_partially_observable:
             true_internal_state = external_obs[-internal_dim:]
        else:
             true_internal_state = env.internal_state.copy()

        estimated_internal_state = torch.zeros(internal_dim, dtype=torch.float32)
        estimator_hidden_state = None

        done = False
        ep_len, total_task_reward, total_homeo_reward, total_intr_reward = 0, 0, 0, 0

        while not done and ep_len < config['env']['horizon']:
            # 1. Construct the full state for the agent and other components
            if is_partially_observable:
                obs_for_agent = np.concatenate([external_obs, estimated_internal_state.detach().numpy()])
                state_for_components = estimated_internal_state.detach().numpy()
            else:
                obs_for_agent = external_obs
                state_for_components = true_internal_state

            # 2. Agent proposes an action
            if total_steps > batch_size:
                if isinstance(agent, MPCAgent):
                    # MPC agent needs a dictionary with obs and internal state
                    agent_input_state = {'obs': external_obs, 'internal': state_for_components}
                    unsafe_action = agent.step(agent_input_state)
                else:
                    # Standard RL agent gets a concatenated state vector
                    unsafe_action = agent.step(obs_for_agent)
            else:
                unsafe_action = env.action_space.sample()

            # 3. Shield projects action to a safe one
            safe_action = shield.project(state_for_components, unsafe_action)

            # 4. Environment steps
            next_external_obs, task_reward, done, info = env.step(safe_action)

            # 5. Get true next internal state
            if is_partially_observable:
                true_next_internal_state = info['internal_state']
            else:
                true_next_internal_state = next_external_obs[-internal_dim:]

            # 6. Estimate next internal state if needed
            if is_partially_observable:
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(external_obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    act_tensor = torch.as_tensor(safe_action, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    predicted_state, next_estimator_hidden_state = state_estimator(obs_tensor, act_tensor, estimator_hidden_state)
                    estimated_next_internal_state = predicted_state.squeeze()
            else:
                estimated_next_internal_state = torch.as_tensor(true_next_internal_state, dtype=torch.float32)
                next_estimator_hidden_state = None

            # 7. Calculate rewards
            homeo_reward = homeostat.reward(estimated_next_internal_state.detach().numpy())

            intrinsic_method = config['rewards']['intrinsic']
            if intrinsic_method == 'rnd':
                intr_reward = intrinsic_reward_module.compute_intrinsic_reward(external_obs)
            elif intrinsic_method == 'empowerment':
                with torch.no_grad():
                    # 1. Get current latent state
                    obs_tensor = torch.as_tensor(external_obs, dtype=torch.float32).unsqueeze(0)
                    z_t = world_model.encoder(obs_tensor)

                    # 2. Sample a k-step action sequence
                    k = intrinsic_reward_module.k
                    action_sequence = torch.stack([
                        torch.from_numpy(env.action_space.sample()) for _ in range(k)
                    ]).float().unsqueeze(0)  # Add batch dimension

                    # 3. Rollout world model to get future latent state
                    current_z = z_t
                    for i in range(k):
                        action_for_rollout = action_sequence[:, i, :]
                        current_z = world_model.transition(current_z, action_for_rollout)
                    z_t_plus_k = current_z

                    # 4. Compute reward
                    intr_reward_tensor = intrinsic_reward_module.compute_reward(z_t, action_sequence, z_t_plus_k)
                    intr_reward = intr_reward_tensor.item()
            else:  # 'surprise'
                intr_reward = world_model.compute_surprise_reward(external_obs, safe_action, next_external_obs)

            total_reward = task_reward + lambda_homeo * homeo_reward + lambda_intr * intr_reward

            # Update MetaLearner (if enabled)
            if meta_learner is not None:
                # Use the true state for a cleaner signal on environmental adaptation
                meta_learner.step(total_reward, true_next_internal_state)

            # 8. Store experience
            viability_label = 1.0 if not (done and info.get('violation', False)) else 0.0
            replay_buffer.store(external_obs, safe_action, unsafe_action, total_reward, next_external_obs, done, true_internal_state, true_next_internal_state, viability_label)

            total_task_reward += task_reward
            total_homeo_reward += homeo_reward
            total_intr_reward += intr_reward

            # 9. Update states for next loop
            external_obs = next_external_obs
            true_internal_state = true_next_internal_state
            estimated_internal_state = estimated_next_internal_state
            estimator_hidden_state = next_estimator_hidden_state

            ep_len += 1
            total_steps += 1

            # Update curriculum parameters
            if scheduler.enabled:
                current_params = scheduler.get_current_values(total_steps)
                lambda_homeo = current_params['lambda_homeo']
                lambda_intr = current_params['lambda_intr']
                env.update_constraints(current_params['constraints'])

            # Switch shield to amortized mode after enough training
            if shield.mode == 'search' and total_steps >= amortize_after_steps:
                print(f"\n--- Switching shield to AMORTIZED mode at step {total_steps} ---")
                shield.mode = 'amortized'

            # Update models
            if total_steps % update_every == 0 and len(replay_buffer) > max(batch_size, sequence_length):
                for _ in range(update_every):
                    # 1. Sample one batch of sequences for all updates to ensure consistency
                    seq_batch = replay_buffer.sample_sequence_batch(batch_size, sequence_length)

                    # Convert all sequence data to tensors
                    obs_seq = torch.as_tensor(seq_batch['obs_seq'], dtype=torch.float32)
                    act_seq = torch.as_tensor(seq_batch['act_seq'], dtype=torch.float32)
                    reward_seq = torch.as_tensor(seq_batch['reward_seq'], dtype=torch.float32)
                    next_obs_seq = torch.as_tensor(seq_batch['next_obs_seq'], dtype=torch.float32)
                    done_seq = torch.as_tensor(seq_batch['done_seq'], dtype=torch.float32)
                    true_internal_state_seq = torch.as_tensor(seq_batch['internal_state_seq'], dtype=torch.float32)
                    true_next_internal_state_seq = torch.as_tensor(seq_batch['next_internal_state_seq'], dtype=torch.float32)
                    unsafe_act_seq = torch.as_tensor(seq_batch['unsafe_action_seq'], dtype=torch.float32)
                    viability_label_seq = torch.as_tensor(seq_batch['viability_label_seq'], dtype=torch.float32)

                    # 2. Get estimated internal states
                    if is_partially_observable:
                        # Create a lagged action sequence for causal prediction
                        act_seq_lagged = torch.cat([torch.zeros_like(act_seq[:, :1]), act_seq[:, :-1]], dim=1)
                        with torch.no_grad():
                            estimated_internal_state_seq, _ = state_estimator(obs_seq, act_seq_lagged)

                        # Train the estimator
                        state_estimator.train_estimator(obs_seq, act_seq_lagged, true_internal_state_seq)
                    else:
                        # If fully observable, the "estimate" is the ground truth
                        estimated_internal_state_seq = true_internal_state_seq

                    # 3. Update world model and intrinsic reward module (using first transition for simplicity)
                    world_model.train_model(obs_seq[:, 0], act_seq[:, 0], next_obs_seq[:, 0])
                    intrinsic_method = config['rewards']['intrinsic']
                    if intrinsic_method == 'rnd':
                        intrinsic_reward_module.train_predictor(obs_seq[:, 0])
                    elif intrinsic_method == 'empowerment':
                        with torch.no_grad():
                            # Generate training data for the discriminator on the fly
                            z_batch = world_model.encoder(obs_seq[:, 0])
                            k = intrinsic_reward_module.k
                            action_sequences_batch = torch.stack([
                                torch.from_numpy(np.array([env.action_space.sample() for _ in range(k)]))
                                for _ in range(batch_size)
                            ]).float()

                            # Rollout the world model to get the batch of future latent states
                            current_z_batch = z_batch
                            for i in range(k):
                                actions_for_rollout = action_sequences_batch[:, i, :]
                                current_z_batch = world_model.transition(current_z_batch, actions_for_rollout)
                            z_future_batch = current_z_batch

                        # Update the empowerment discriminator
                        intrinsic_reward_module.update(z_batch.detach(), action_sequences_batch, z_future_batch.detach())


                    # 4. Update safety models using ESTIMATED states for robustness
                    # Flatten sequences for batch training
                    flat_est_internal = estimated_internal_state_seq.view(-1, internal_dim)
                    flat_true_next_internal = true_next_internal_state_seq.view(-1, internal_dim)
                    flat_act = act_seq.view(-1, env.action_dim)
                    flat_unsafe_act = unsafe_act_seq.view(-1, env.action_dim)
                    flat_viability_label = viability_label_seq.view(-1)

                    internal_model.train_model(flat_est_internal, flat_act, flat_true_next_internal)

                    if demo_buffer and len(demo_buffer) > 0:
                        viability_approximator.train_on_demonstrations(demo_buffer, batch_size)
                    viability_approximator.train_model(flat_est_internal, flat_viability_label)

                    safety_network.train_network(
                        unsafe_actions=flat_unsafe_act,
                        safe_actions=flat_act,
                        internal_states=flat_est_internal
                    )

                    # 5. Construct a consistent batch for the agent from the LAST transition
                    agent_batch = {}
                    last_obs = torch.cat([obs_seq[:, -1], estimated_internal_state_seq[:, -1]], dim=-1)
                    # Use ground truth for next state to stabilize critic learning
                    last_next_obs = torch.cat([next_obs_seq[:, -1], true_next_internal_state_seq[:, -1]], dim=-1)

                    agent_batch['obs'] = last_obs.detach().numpy()
                    agent_batch['next_obs'] = last_next_obs.detach().numpy()
                    agent_batch['action'] = act_seq[:, -1].detach().numpy()
                    agent_batch['reward'] = reward_seq[:, -1].detach().numpy()
                    agent_batch['done'] = done_seq[:, -1].detach().numpy()

                    agent.learn(data=agent_batch)

        # Handle end-of-episode updates for the meta-learner
        if meta_learner is not None:
            meta_learner.episode_end()
            homeostat.mu = meta_learner.get_setpoints()

        print(f"Episode {episode + 1}: Steps = {ep_len}, Task Reward = {total_task_reward:.2f}, "
              f"Homeo Reward = {total_homeo_reward:.2f}, Intr Reward = {total_intr_reward:.2f}")

if __name__ == "__main__":
    main()