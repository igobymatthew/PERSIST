import numpy as np
from components.reach_avoid_mpc import ReachAvoidMPC

class MPCAgent:
    def __init__(self, latent_world_model, internal_model, viability_approximator, action_space, mpc_config):
        """
        Initializes the MPC-based agent.

        This agent uses a Reach-Avoid MPC component to select actions, instead of a
        traditional policy network.

        Args:
            latent_world_model: The model for predicting future latent states.
            internal_model: The model for predicting future internal states.
            viability_approximator: The model for estimating the safety margin.
            action_space: The environment's action space.
            mpc_config (dict): A dictionary containing hyperparameters for the MPC component.
        """
        print("Initializing MPC Agent...")
        self.mpc = ReachAvoidMPC(
            latent_world_model=latent_world_model,
            internal_model=internal_model,
            viability_approximator=viability_approximator,
            action_space=action_space,
            horizon=mpc_config.get('horizon', 12),
            num_candidates=mpc_config.get('num_candidates', 1000),
            top_k=mpc_config.get('top_k', 100),
            iterations=mpc_config.get('iterations', 10)
        )
        self.action_space = action_space

    def step(self, state, deterministic=True):
        """
        Selects an action by planning with the MPC component.

        Args:
            state (dict): The current state, containing 'obs' and 'internal'.
            deterministic (bool): Ignored, as MPC is inherently deterministic.

        Returns:
            np.ndarray: The selected action.
        """
        # The MPC planner takes care of selecting the best action
        action = self.mpc.plan_action(state)
        return np.clip(action, self.action_space.low, self.action_space.high)

    def learn(self, data):
        """
        The MPC agent does not learn in the same way as a policy-gradient agent.
        The underlying models (world model, internal model, etc.) are trained
        separately in the main training loop. This method is a no-op.
        """
        pass

    def get_state(self):
        return {}

    def load_state(self, state):
        # The MPC agent has no trainable parameters to restore.
        return

    def get_optimizer_state(self):
        return {}

    def load_optimizer_state(self, state):
        return

    def get_optimizers(self):
        return {}
