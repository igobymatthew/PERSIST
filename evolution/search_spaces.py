# persist/evolution/search_spaces.py

def ppo_small_net_v1():
    """
    Defines a search space for PPO hyperparameters and a small network.
    """
    return {
        "lr": ("log_uniform", 1e-5, 1e-3),
        "gamma": ("uniform", 0.9, 0.999),
        "net_width": ("choice", [64, 128, 256]),
        "net_depth": ("choice", [2, 3, 4]),
    }