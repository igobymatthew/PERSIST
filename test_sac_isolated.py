import torch
import numpy as np
from agents.sac import SAC

def main():
    print("--- Starting Isolated SAC Test ---")
    try:
        obs_dim = 10
        act_dim = 2
        batch_size = 32

        print("Initializing SAC agent...")
        agent = SAC(obs_dim, act_dim)
        print("✅ SAC agent initialized.")

        print("\nCreating dummy data...")
        dummy_data = {
            'obs': np.random.rand(batch_size, obs_dim),
            'action': np.random.rand(batch_size, act_dim),
            'reward': np.random.rand(batch_size),
            'next_obs': np.random.rand(batch_size, obs_dim),
            'done': np.random.randint(0, 2, batch_size)
        }
        print("✅ Dummy data created.")

        print("\nRunning a single update...")
        agent.update(dummy_data)
        print("✅ SAC update completed successfully.")

        print("\n--- ✅ Isolated SAC Test Successful! ---")

    except Exception as e:
        import sys
        print(f"\n--- ❌ An error occurred during the isolated test ---", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()