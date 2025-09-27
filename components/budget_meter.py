import torch

class BudgetMeter:
    """
    Tracks a budget for a resource (e.g., computation, energy) and provides penalties.
    """
    def __init__(self, budget_config):
        """
        Initializes the BudgetMeter.

        Args:
            budget_config (dict): Configuration for the budget, including:
                - initial_budget (float): The starting budget for each episode.
                - penalty (float): The penalty applied when the budget is exhausted.
        """
        self.initial_budget = budget_config.get("initial_budget", 1.0)
        self.penalty = budget_config.get("penalty", -1.0)
        self.current_budget = self.initial_budget

    def reset(self):
        """
        Resets the budget to its initial value.
        """
        self.current_budget = self.initial_budget

    def decrement(self, amount):
        """
        Decrements the budget by a given amount.

        Args:
            amount (float): The amount to decrement the budget by.
        """
        self.current_budget -= amount

    def is_exhausted(self):
        """
        Checks if the budget has been exhausted.

        Returns:
            bool: True if the budget is exhausted, False otherwise.
        """
        return self.current_budget <= 0

    def get_penalty(self):
        """
        Returns the penalty for exhausting the budget.

        Returns:
            float: The penalty value.
        """
        return self.penalty

    def get_state(self):
        """
        Returns the current state of the budget.

        Returns:
            dict: A dictionary containing the current budget.
        """
        return {"current_budget": self.current_budget}