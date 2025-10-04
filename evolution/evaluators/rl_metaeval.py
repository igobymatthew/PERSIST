# persist/evolution/evaluators/rl_metaeval.py

def train_and_measure(cfg):
    """
    A placeholder for a function that trains an RL agent and returns metrics.
    """
    # In a real scenario, this would involve a complex training loop.
    # Here, we return dummy values based on the config.
    ep_return = 100 - (cfg["net_width"] / 100) - (cfg["net_depth"] * 10)
    violations = 10 / cfg["lr"] * 1e-5
    latency = cfg["net_width"] * 0.1
    return ep_return, violations, latency


def eval_rl_individual(ind) -> dict:
    """
    Evaluates an individual by training an RL agent with its genes as config.
    """
    cfg = ind["genes"]  # net width/depth, lr, entropy, γ, λ, etc.
    # train PPO for N steps (inner loop), collect metrics
    ep_return, violations, latency = train_and_measure(cfg)
    return {"neg_return": -ep_return, "viol": violations, "latency": latency}