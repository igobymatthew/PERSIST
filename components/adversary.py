import torch
import torch.nn.functional as F

class Adversary:
    """
    Generates adversarial perturbations for observations.
    Uses Projected Gradient Descent (PGD) to create adversarial examples.
    """
    def __init__(self, epsilon=0.05, alpha=0.01, num_iter=10):
        """
        Initializes the PGD adversary.

        Args:
            epsilon (float): The maximum perturbation magnitude.
            alpha (float): The step size for each iteration.
            num_iter (int): The number of PGD iterations.
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter

    def perturb(self, model, obs, action, target):
        """
        Generates an adversarial observation using PGD.

        Args:
            model (torch.nn.Module): The model to attack (e.g., the critic).
            obs (torch.Tensor): The original observation.
            action (torch.Tensor): The action taken in the state.
            target (torch.Tensor): The target value for the critic.

        Returns:
            torch.Tensor: The perturbed observation.
        """
        obs_adv = obs.clone().detach().requires_grad_(True)
        original_obs = obs.clone().detach()

        for _ in range(self.num_iter):
            # Forward pass to get model output
            q_value = model(obs_adv, action)
            loss = F.mse_loss(q_value, target)

            # Backward pass to get gradients
            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                # PGD update
                grad = obs_adv.grad
                obs_adv = obs_adv + self.alpha * grad.sign()

                # Project back into epsilon-ball
                perturbation = torch.clamp(obs_adv - original_obs, -self.epsilon, self.epsilon)
                obs_adv = original_obs + perturbation

                # Ensure the perturbed observation is within valid range (e.g., [0, 1])
                obs_adv = torch.clamp(obs_adv, 0, 1)

        return obs_adv.detach()