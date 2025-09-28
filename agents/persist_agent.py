import numpy as np
from .sac import SAC
from .cvar_sac import CVAR_SAC

class PersistAgent:
    def __init__(self, obs_dim, act_dim, act_limit, risk_sensitive_config=None):
        if risk_sensitive_config and risk_sensitive_config.get('enabled', False):
            print("Initializing Risk-Sensitive Agent (CVAR-SAC)...")
            self.policy = CVAR_SAC(
                obs_dim=obs_dim,
                act_dim=act_dim,
                n_quantiles=risk_sensitive_config.get('n_quantiles', 32),
                tau=risk_sensitive_config.get('tau', 0.1)
            )
        else:
            print("Initializing Standard Agent (SAC)...")
            self.policy = SAC(obs_dim, act_dim)

        self.act_limit = act_limit

    def step(self, s, deterministic=False):
        action = self.policy.get_action(s, deterministic)
        return np.clip(action, -self.act_limit, self.act_limit)

    def learn(self, data, ewc_penalty=0.0):
        # Pass the penalty to the underlying policy's update method
        # This works for SAC. If CVAR_SAC needs it, it would need a similar change.
        if isinstance(self.policy, SAC):
            return self.policy.learn(data, ewc_penalty=ewc_penalty)
        else:
            # For now, other policies don't use the penalty
            return self.policy.update(data)

    def to(self, device):
        self.policy.to(device)
        return self