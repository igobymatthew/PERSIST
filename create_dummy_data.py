import numpy as np

# Create dummy data to simulate expert demonstrations
# In a real scenario, this data would be collected from an expert policy.
num_demonstrations = 1000
observation_dim = 10  # Example observation dimension
internal_dim = 3    # energy, temp, integrity

# Simulate safe states
observations = np.random.rand(num_demonstrations, observation_dim).astype(np.float32)
# Expert internal states are kept within safe bounds
internal_states = np.random.uniform(
    low=[0.3, 0.4, 0.7],  # energy, temp, integrity (well within safe ranges)
    high=[0.9, 0.6, 1.0],
    size=(num_demonstrations, internal_dim)
).astype(np.float32)

# Save the demonstrations to a file
np.savez(
    'demonstrations/expert_demonstrations.npz',
    observations=observations,
    internal_states=internal_states
)

print("Dummy demonstration data created at 'demonstrations/expert_demonstrations.npz'")