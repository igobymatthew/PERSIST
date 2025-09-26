import yaml
import numpy as np

from environments.grid_life import GridLifeEnv
from agents.persist_agent import PersistAgent
from components.homeostat import Homeostat

def main():
    # Load configuration
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Initialize components
    env = GridLifeEnv(config)
    agent = PersistAgent(action_space=env.action_space)
    homeostat = Homeostat(
        mu=config['internal_state']['mu'],
        w=config['internal_state']['w']
    )

    lambda_homeo = config['rewards']['lambda_homeo']

    # Training loop
    num_episodes = 10
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_task_reward = 0
        total_homeo_reward = 0
        step = 0

        while not done and step < config['env']['horizon']:
            # Agent takes an action
            action = agent.step(obs)

            # Environment steps
            next_obs, task_reward, done, _ = env.step(action)

            # Extract internal state from observation
            internal_state = next_obs[-len(config['internal_state']['dims']):]

            # Calculate homeostatic reward
            homeo_reward = homeostat.reward(internal_state)

            # Calculate total reward (for this phase, we ignore intrinsic reward)
            total_reward = task_reward + lambda_homeo * homeo_reward

            total_task_reward += task_reward
            total_homeo_reward += homeo_reward

            obs = next_obs
            step += 1

        print(f"Episode {episode + 1}:")
        print(f"  Steps = {step}")
        print(f"  Total Task Reward = {total_task_reward:.2f}")
        print(f"  Total Homeostatic Reward = {total_homeo_reward:.2f}")
        print("-" * 20)

if __name__ == "__main__":
    main()