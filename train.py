import yaml
import numpy as np

from environments.grid_life import GridLifeEnv
from agents.persist_agent import PersistAgent
from components.homeostat import Homeostat
from components.replay_buffer import ReplayBuffer
from components.world_model import WorldModel
from components.internal_model import InternalModel
from components.viability_approximator import ViabilityApproximator
from components.shield import Shield

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
    internal_dim = len(config['internal_state']['dims'])
    print("✅ Environment initialized.")

    print("\nInitializing agent...")
    agent = PersistAgent(
        obs_dim=env.observation_space_dim,
        act_dim=env.action_dim,
        act_limit=env.act_limit
    )
    print("✅ Agent initialized.")

    print("\nInitializing homeostat...")
    homeostat = Homeostat(
        mu=config['internal_state']['mu'],
        w=config['internal_state']['w']
    )
    print("✅ Homeostat initialized.")

    print("\nInitializing world model...")
    world_model = WorldModel(
        obs_dim=env.observation_space_dim,
        act_dim=env.action_dim
    )
    print("✅ World model initialized.")

    print("\nInitializing internal model...")
    internal_model = InternalModel(
        internal_dim=internal_dim,
        act_dim=env.action_dim
    )
    print("✅ Internal model initialized.")

    print("\nInitializing viability approximator...")
    viability_approximator = ViabilityApproximator(internal_dim=internal_dim)
    print("✅ Viability approximator initialized.")

    print("\nInitializing safety shield...")
    shield = Shield(
        internal_model=internal_model,
        viability_approximator=viability_approximator,
        action_space=env.action_space, # Pass the gym action space
        conf=config['viability']['shield']['conf']
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

    print("\n--- ✅ All Components Initialized ---")

    lambda_homeo = config['rewards']['lambda_homeo']
    lambda_intr = config['rewards']['lambda_intr']
    batch_size = config['train']['batch_size']
    update_every = config['train']['update_every']
    total_steps = 0

    num_episodes = 500
    for episode in range(num_episodes):
        obs = env.reset()
        internal_state = obs[-internal_dim:]
        done = False
        ep_len, total_task_reward, total_homeo_reward, total_intr_reward = 0, 0, 0, 0

        while not done and ep_len < config['env']['horizon']:
            # Agent proposes an action
            if total_steps > batch_size:
                action = agent.step(obs)
            else:
                action = env.action_space.sample()

            # Shield projects action to a safe one
            safe_action = shield.project(internal_state, action)

            # Environment steps with the safe action
            next_obs, task_reward, done, info = env.step(safe_action)
            next_internal_state = next_obs[-internal_dim:]

            # Determine viability label (1 for safe, 0 for unsafe)
            # A simple heuristic: if the episode ends due to a constraint violation, the state was not viable.
            viability_label = 1.0 if not (done and info.get('violation', False)) else 0.0

            # Calculate rewards
            homeo_reward = homeostat.reward(next_internal_state)
            intr_reward = world_model.compute_surprise_reward(obs, safe_action, next_obs)
            total_reward = task_reward + lambda_homeo * homeo_reward + lambda_intr * intr_reward

            # Store experience
            replay_buffer.store(obs, safe_action, total_reward, next_obs, done, internal_state, next_internal_state, viability_label)

            total_task_reward += task_reward
            total_homeo_reward += homeo_reward
            total_intr_reward += intr_reward

            obs = next_obs
            internal_state = next_internal_state
            ep_len += 1
            total_steps += 1

            # Update models
            if total_steps % update_every == 0 and len(replay_buffer) > batch_size:
                for _ in range(update_every):
                    batch = replay_buffer.sample_batch(batch_size)
                    # Update world model
                    world_model.train_model(batch['obs'], batch['action'], batch['next_obs'])
                    # Update internal model
                    internal_model.train_model(batch['internal_state'], batch['action'], batch['next_internal_state'])
                    # Update viability approximator
                    viability_approximator.train_model(batch['next_internal_state'], batch['viability_label'])
                    # Update agent
                    agent.learn(data=batch)

        print(f"Episode {episode + 1}: Steps = {ep_len}, Task Reward = {total_task_reward:.2f}, "
              f"Homeo Reward = {total_homeo_reward:.2f}, Intr Reward = {total_intr_reward:.2f}")

if __name__ == "__main__":
    main()