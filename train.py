import yaml
import numpy as np

from environments.grid_life import GridLifeEnv
from agents.persist_agent import PersistAgent
from components.homeostat import Homeostat
from components.replay_buffer import ReplayBuffer
from components.world_model import WorldModel

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

    print("\nInitializing replay buffer...")
    replay_buffer = ReplayBuffer(
        capacity=10000,
        obs_dim=env.observation_space_dim,
        action_dim=env.action_dim
    )
    print("✅ Replay buffer initialized.")

    print("\n--- ✅ All Components Initialized ---")

    lambda_homeo = config['rewards']['lambda_homeo']
    lambda_intr = config['rewards']['lambda_intr']
    batch_size = config['train']['batch_size']
    update_every = config['train']['update_every']
    total_steps = 0

    # Training loop
    num_episodes = 500
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_task_reward = 0
        total_homeo_reward = 0
        total_intr_reward = 0
        ep_len = 0

        while not done and ep_len < config['env']['horizon']:
            # Agent takes an action
            if total_steps > batch_size:
                action = agent.step(obs)
            else:
                action = np.random.rand(env.action_dim) * 2 - 1

            # Environment steps
            next_obs, task_reward, done, _ = env.step(action)

            # Extract internal state from observation
            internal_state = next_obs[-len(config['internal_state']['dims']):]

            # Calculate homeostatic reward
            homeo_reward = homeostat.reward(internal_state)

            # Calculate intrinsic reward
            intr_reward = world_model.compute_surprise_reward(obs, action, next_obs)

            # Calculate total reward
            total_reward = task_reward + lambda_homeo * homeo_reward + lambda_intr * intr_reward

            # Store experience
            replay_buffer.store(obs, action, total_reward, next_obs, done)

            total_task_reward += task_reward
            total_homeo_reward += homeo_reward
            total_intr_reward += intr_reward

            obs = next_obs
            ep_len += 1
            total_steps += 1

            # Update models
            if total_steps % update_every == 0 and len(replay_buffer) > batch_size:
                for _ in range(update_every):
                    batch = replay_buffer.sample_batch(batch_size)
                    # Update world model
                    world_model.train_model(batch['obs'], batch['action'], batch['next_obs'])
                    # Update agent
                    agent.learn(data=batch)

        print(f"Episode {episode + 1}:")
        print(f"  Steps = {ep_len}")
        print(f"  Total Task Reward = {total_task_reward:.2f}")
        print(f"  Total Homeostatic Reward = {total_homeo_reward:.2f}")
        print(f"  Total Intrinsic Reward = {total_intr_reward:.2f}")
        print("-" * 20)

if __name__ == "__main__":
    main()