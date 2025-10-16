import sys

print("--- Starting Import Test ---")
try:
    print("Importing yaml...")
    import yaml  # noqa: F401
    print("✅ yaml imported successfully.")

    print("\nImporting numpy...")
    import numpy as np  # noqa: F401
    print("✅ numpy imported successfully.")

    print("\nImporting torch...")
    import torch  # noqa: F401
    print("✅ torch imported successfully.")

    print("\nImporting GridLifeEnv...")
    from environments.grid_life import GridLifeEnv  # noqa: F401
    print("✅ GridLifeEnv imported successfully.")

    print("\nImporting Homeostat...")
    from components.homeostat import Homeostat  # noqa: F401
    print("✅ Homeostat imported successfully.")

    print("\nImporting ReplayBuffer...")
    from components.replay_buffer import ReplayBuffer  # noqa: F401
    print("✅ ReplayBuffer imported successfully.")

    print("\nImporting SAC...")
    from agents.sac import SAC  # noqa: F401
    print("✅ SAC imported successfully.")

    print("\nImporting PersistAgent...")
    from agents.persist_agent import PersistAgent  # noqa: F401
    print("✅ PersistAgent imported successfully.")

    print("\nImporting SafetyNetwork...")
    from components.safety_network import SafetyNetwork  # noqa: F401
    print("✅ SafetyNetwork imported successfully.")

    print("\n--- ✅ All Imports Successful! ---")

except Exception as e:
    print("\n--- ❌ An error occurred during import ---", file=sys.stderr)
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
