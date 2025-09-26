import numpy as np

class PersistAgent:
    def __init__(self, action_space):
        # The following components will be added in later phases
        # self.policy = ...
        # self.shield = ...
        # self.internal_model = ...
        # self.viability = ...
        self.action_space = action_space

    def step(self, s):
        # For now, the agent takes a random action.
        # The full implementation will involve the policy and shield.
        # a = self.policy.sample(s)
        # a = self.shield.project(a, s, self.internal_model, self.viability)
        return np.random.randint(self.action_space)