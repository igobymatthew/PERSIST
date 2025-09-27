import pytest
import numpy as np
import torch
from components.homeostat import Homeostat

def test_homeostat_reward_calculation():
    """
    Tests that the Homeostat component calculates the reward correctly.
    """
    # 1. Setup
    mu = np.array([0.5, 0.8])
    w = np.array([1.0, 2.0])
    homeostat = Homeostat(mu=mu, w=w)

    # 2. Test case 1: State is exactly at the setpoint (mu)
    x_at_setpoint = np.array([0.5, 0.8])
    reward_at_setpoint = homeostat.reward(x_at_setpoint)
    # Expected reward should be 0, as there is no deviation
    expected_reward_at_setpoint = 0.0
    assert np.isclose(reward_at_setpoint, expected_reward_at_setpoint), \
        f"Reward at setpoint should be {expected_reward_at_setpoint}, but got {reward_at_setpoint}"

    # 3. Test case 2: State deviates from the setpoint
    x_deviated = np.array([0.6, 0.7])
    reward_deviated = homeostat.reward(x_deviated)
    # Expected reward = -[ w[0]*(x[0]-mu[0])^2 + w[1]*(x[1]-mu[1])^2 ]
    #                = -[ 1.0*(0.6-0.5)^2 + 2.0*(0.7-0.8)^2 ]
    #                = -[ 1.0*0.01 + 2.0*0.01 ]
    #                = -[ 0.01 + 0.02 ] = -0.03
    expected_reward_deviated = -0.03
    assert np.isclose(reward_deviated, expected_reward_deviated), \
        f"Reward for deviated state should be {expected_reward_deviated}, but got {reward_deviated}"

    # 4. Test case 3: Test with a batch of states (e.g., from a replay buffer)
    x_batch = np.array([
        [0.5, 0.8],  # At setpoint
        [0.6, 0.7],  # Deviated
        [0.4, 0.9]   # Deviated symmetrically
    ])
    reward_batch = homeostat.reward(x_batch)
    expected_reward_batch = np.array([
        0.0,
        -0.03,
        -0.03  # Same deviation squared, so same reward
    ])
    assert np.allclose(reward_batch, expected_reward_batch), \
        f"Reward for batch of states was incorrect. Expected {expected_reward_batch}, got {reward_batch}"

from components.viability_approximator import ViabilityApproximator

def test_viability_approximator_training_and_prediction():
    """
    Tests that the ViabilityApproximator can be trained and make predictions.
    """
    # 1. Setup
    internal_dim = 3
    approximator = ViabilityApproximator(internal_dim=internal_dim, hidden_dim=16)

    # 2. Create dummy training data
    # Let's say states with sum > 1.5 are viable (label 1.0)
    # and states with sum <= 1.5 are not (label 0.0)
    viable_states = torch.tensor([
        [0.8, 0.8, 0.1],
        [0.5, 0.6, 0.5],
        [0.9, 0.9, 0.9]
    ], dtype=torch.float32)
    non_viable_states = torch.tensor([
        [0.1, 0.2, 0.3],
        [0.5, 0.5, 0.5],
        [0.2, 0.4, 0.1]
    ], dtype=torch.float32)
    states = torch.cat([viable_states, non_viable_states], dim=0)
    labels = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    # 3. Train the model
    # Increased iterations and learning rate to make test less flaky.
    approximator.optimizer = torch.optim.Adam(approximator.parameters(), lr=5e-3)
    initial_loss = approximator.train_model(states, labels)
    for _ in range(200):
        loss = approximator.train_model(states, labels)
    final_loss = loss

    # Check that the loss has decreased
    assert final_loss < initial_loss, f"Loss should decrease after training, but went from {initial_loss} to {final_loss}"

    # 4. Test predictions
    approximator.eval()  # Set to evaluation mode
    with torch.no_grad():
        # Test with states similar to the training data
        test_viable_state = torch.tensor([[0.7, 0.7, 0.7]], dtype=torch.float32) # sum = 2.1
        test_non_viable_state = torch.tensor([[0.3, 0.3, 0.3]], dtype=torch.float32) # sum = 0.9

        # The model outputs a probability (0 to 1). We can treat > 0.5 as viable.
        viable_prob = approximator(test_viable_state)
        non_viable_prob = approximator(test_non_viable_state)

        # The probability for the viable state should be > 0.5
        # The probability for the non-viable state should be < 0.5
        assert viable_prob.item() > 0.5, f"Expected > 0.5 for viable state, but got {viable_prob.item()}"
        assert non_viable_prob.item() < 0.5, f"Expected < 0.5 for non-viable state, but got {non_viable_prob.item()}"

from components.internal_model import InternalModel

def test_internal_model_training_and_prediction():
    """
    Tests that the InternalModel can be trained to predict the next internal state.
    """
    # 1. Setup
    internal_dim = 2
    act_dim = 1
    model = InternalModel(internal_dim=internal_dim, act_dim=act_dim, hidden_dim=64)

    # 2. Create dummy training data
    # Let's define a simple dynamic: next_state = current_state + action
    # current_states: [energy, temp]
    # actions: [delta_energy]
    # next_states: [energy + delta_energy, temp] (temp is unaffected)
    current_states = torch.tensor([
        [0.5, 0.5],
        [0.8, 0.3],
        [0.2, 0.9]
    ], dtype=torch.float32)
    actions = torch.tensor([[0.1], [-0.1], [0.05]], dtype=torch.float32)
    next_states_true = torch.tensor([
        [0.6, 0.5],
        [0.7, 0.3],
        [0.25, 0.9]
    ], dtype=torch.float32)

    # 3. Train the model
    initial_loss = model.train_model(current_states, actions, next_states_true)
    for _ in range(500): # Increased training iterations for robustness to prevent flaky tests
        loss = model.train_model(current_states, actions, next_states_true)
    final_loss = loss

    # Check that the loss has decreased
    assert final_loss < initial_loss, f"Loss should decrease after training. Went from {initial_loss} to {final_loss}"

    # 4. Test prediction
    model.eval() # Set the model to evaluation mode
    with torch.no_grad():
        test_state = torch.tensor([0.4, 0.6], dtype=torch.float32)
        test_action = torch.tensor([0.2], dtype=torch.float32)
        predicted_next_state = model.predict_next(test_state, test_action)

        expected_next_state = torch.tensor([0.6, 0.6], dtype=torch.float32)

        # Check if the prediction is reasonably close to the true next state
        # The model's predict_next returns a numpy array, so we use np.allclose
        assert np.allclose(predicted_next_state, expected_next_state.numpy(), atol=0.1), \
            f"Predicted next state {predicted_next_state} is not close to expected {expected_next_state.numpy()}"

from components.budget_meter import BudgetMeter

def test_budget_meter():
    """
    Tests the functionality of the BudgetMeter component.
    """
    # 1. Setup
    config = {
        "initial_budget": 1.0,
        "penalty": -10.0
    }
    budget_meter = BudgetMeter(budget_config=config)

    # 2. Test initialization
    assert budget_meter.current_budget == 1.0, "Initial budget is incorrect."
    assert not budget_meter.is_exhausted(), "Budget should not be exhausted initially."

    # 3. Test decrement
    budget_meter.decrement(0.3)
    assert np.isclose(budget_meter.current_budget, 0.7), "Decrement did not work as expected."
    assert not budget_meter.is_exhausted(), "Budget should not be exhausted after a small decrement."

    # 4. Test exhaustion
    budget_meter.decrement(0.7)
    assert np.isclose(budget_meter.current_budget, 0.0), "Budget should be zero."
    assert budget_meter.is_exhausted(), "Budget should be exhausted when current_budget is zero."

    # 5. Test penalty
    assert budget_meter.get_penalty() == -10.0, "Penalty amount is incorrect."

    # 6. Test reset
    budget_meter.reset()
    assert budget_meter.current_budget == 1.0, "Reset did not restore the initial budget."
    assert not budget_meter.is_exhausted(), "Budget should not be exhausted after reset."

    # 7. Test over-exhaustion
    budget_meter.decrement(1.1)
    assert budget_meter.current_budget < 0, "Budget should be negative after over-exhaustion."
    assert budget_meter.is_exhausted(), "Budget should be exhausted when over-exhausted."