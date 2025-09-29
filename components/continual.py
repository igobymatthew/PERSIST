import torch
import torch.nn as nn
import numpy as np

class ContinualLearningManager:
    """
    Manages the continual learning process using Elastic Weight Consolidation (EWC).

    EWC protects weights important for previous tasks from being overwritten
    when learning a new task. It does this by adding a quadratic penalty
    to the loss function, where the penalty is proportional to the weight's
    estimated importance.
    """
    def __init__(self, agent, ewc_lambda=1.0, device='cpu'):
        """
        Initializes the ContinualLearningManager.

        Args:
            agent (PersistAgent): The agent whose weights will be protected.
            ewc_lambda (float): The hyperparameter that controls the strength
                                of the EWC penalty.
            device (str): The device to run computations on ('cpu' or 'cuda').
        """
        self.agent = agent
        self.actor = self._resolve_actor(agent)
        self.ewc_lambda = ewc_lambda
        self.device = device
        self.fisher_information = {}
        self.optimal_params = {}
        self.task_count = 0

    def consolidate(self, rehearsal_buffer):
        """
        Computes and stores the Fisher Information Matrix and optimal parameters
        for the current task. This should be called periodically.

        Args:
            rehearsal_buffer (RehearsalBuffer): A buffer containing a sample
                                                 of past states.
        """
        if len(rehearsal_buffer) == 0:
            print("--- Skipping EWC consolidation: Rehearsal buffer is empty. ---")
            return

        print("--- Consolidating task knowledge for EWC ---")

        # 1. Store the current optimal parameters for the actor network
        self.optimal_params = {name: p.data.clone() for name, p in self.actor.named_parameters() if p.requires_grad}

        # 2. Compute the Fisher Information Matrix
        self._compute_fisher(rehearsal_buffer)

        self.task_count += 1
        print(f"--- Consolidation complete. Fisher matrix updated. ---")

    def _compute_fisher(self, rehearsal_buffer):
        """
        Computes the diagonal of the Fisher Information Matrix using states
        from the rehearsal buffer.
        """
        fisher_new = {name: torch.zeros_like(p.data) for name, p in self.actor.named_parameters() if p.requires_grad}

        self.actor.train()

        # Sample states from the rehearsal buffer to estimate Fisher information
        states_sample = rehearsal_buffer.sample(batch_size=256) # Use a fixed batch for estimation
        states = torch.from_numpy(np.array(states_sample)).float().to(self.device)

        # Zero gradients before computation
        self.actor.zero_grad()

        # Get the action distribution from the policy for the sampled states
        dist = self.actor(states)

        # Use the log probability of actions sampled from the distribution.
        # The Fisher Information is the expectation of the squared gradient of the log-likelihood.
        # We approximate this by taking the mean over a batch of sampled actions.
        log_prob = dist.log_prob(dist.sample()).sum()
        log_prob.backward()

        for name, param in self.actor.named_parameters():
            if param.grad is not None:
                fisher_new[name] = param.grad.data.pow(2)

        # Update the stored Fisher information.
        # A common approach is to add the new Fisher information to the old one.
        # This way, importance accumulates over time.
        for name in fisher_new:
            if name in self.fisher_information:
                self.fisher_information[name] += fisher_new[name]
            else:
                self.fisher_information[name] = fisher_new[name]

    def penalty(self):
        """
        Calculates the EWC penalty to be added to the agent's loss.
        This should be called during the agent's update step.

        Returns:
            torch.Tensor: The EWC penalty term, or zero if no tasks have been
                          consolidated yet.
        """
        if not self.fisher_information:
            return torch.tensor(0.0, device=self.device)

        penalty = 0.0
        for name, param in self.actor.named_parameters():
            if name in self.fisher_information:
                fisher = self.fisher_information[name]
                opt_param = self.optimal_params[name]
                penalty += (fisher * (param - opt_param).pow(2)).sum()

        return self.ewc_lambda * penalty

    def _resolve_actor(self, agent) -> nn.Module:
        if hasattr(agent, 'actor') and isinstance(agent.actor, nn.Module):
            return agent.actor
        if hasattr(agent, 'policy') and hasattr(agent.policy, 'actor') and isinstance(agent.policy.actor, nn.Module):
            return agent.policy.actor
        raise ValueError("ContinualLearningManager requires an agent with an actor network.")

