import torch
import numpy as np
import argparse
import os
import sys

# Add project root to the Python path to allow for package imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.viability_approximator import ViabilityApproximator
from torch.utils.data import TensorDataset, DataLoader

def distill_viability_model(
    data_path,
    model_save_path,
    internal_dim,
    hidden_dim=256,
    lr=1e-3,
    epochs=50,
    batch_size=1024,
    device="cpu",
):
    """
    Trains a ViabilityApproximator model on a pre-computed viable set.

    This process "distills" the knowledge from an offline, computationally
    expensive analysis (like HJ reachability) into a fast neural network model.

    Args:
        data_path (str): Path to the .npz file containing the viable set data.
        model_save_path (str): Path to save the trained model's weights.
        internal_dim (int): The dimension of the internal state space.
        hidden_dim (int): The hidden dimension size for the model.
        lr (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        device (str): The device to train on ('cpu' or 'cuda').
    """
    print(f"Loading viable set data from {data_path}...")
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}.")
        print("Please run 'python tools/hj_reachability/compute_viability.py' first.")
        return

    data = np.load(data_path)
    states = data["states"].astype(np.float32)
    # Labels should be float32 for BCELoss and flattened to match states
    labels = data["viable_mask"].ravel().astype(np.float32)

    print(f"Loaded {len(states)} state-label pairs.")

    # Create a PyTorch DataLoader for batching
    dataset = TensorDataset(torch.from_numpy(states), torch.from_numpy(labels))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Initializing ViabilityApproximator model on device '{device}'...")
    model = ViabilityApproximator(internal_dim=internal_dim, hidden_dim=hidden_dim, lr=lr)
    model.to(device)
    model.train()  # Set the model to training mode

    print(f"Starting distillation training for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_states, batch_labels in loader:
            batch_states, batch_labels = batch_states.to(device), batch_labels.to(device)
            loss = model.train_model(batch_states, batch_labels)
            total_loss += loss

        avg_loss = total_loss / len(loader)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")

    # Ensure the directory for the saved model exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Save the trained model's state dictionary
    torch.save(model.state_dict(), model_save_path)
    print(f"Distillation complete. Model saved to {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill a Viability Approximator model.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/viable_set.npz",
        help="Path to the pre-computed viable set data.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/viability_approximator_distilled.pth",
        help="Path to save the trained model weights.",
    )
    parser.add_argument("--internal-dim", type=int, default=2, help="Dimension of internal state.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension of the model.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Training batch size.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use ('cpu' or 'cuda').")

    args = parser.parse_args()

    distill_viability_model(
        data_path=args.data_path,
        model_save_path=args.save_path,
        internal_dim=args.internal_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )