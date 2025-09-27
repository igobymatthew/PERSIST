import sys

print("--- Starting Import Test ---")
try:
    print("Importing yaml...")
    import yaml
    print("✅ yaml imported successfully.")

    print("\nImporting numpy...")
    import numpy as np
    print("✅ numpy imported successfully.")

    print("\nImporting torch...")
    import torch
    print("✅ torch imported successfully.")

    print("\nImporting GridLifeEnv...")
    from environments.grid_life import GridLifeEnv
    print("✅ GridLifeEnv imported successfully.")

    print("\nImporting Homeostat...")
    from components.homeostat import Homeostat
    print("✅ Homeostat imported successfully.")

    print("\nImporting ReplayBuffer...")
    from components.replay_buffer import ReplayBuffer
    print("✅ ReplayBuffer imported successfully.")

    print("\nImporting SAC...")
    from agents.sac import SAC
    print("✅ SAC imported successfully.")

    print("\nImporting PersistAgent...")
    from agents.persist_agent import PersistAgent
    print("✅ PersistAgent imported successfully.")

    print("\nImporting SafetyNetwork...")
    from components.safety_network import SafetyNetwork
    print("✅ SafetyNetwork imported successfully.")

    print("\n--- ✅ All Imports Successful! ---")

except Exception as e:
    print(f"\n--- ❌ An error occurred during import ---", file=sys.stderr)
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)