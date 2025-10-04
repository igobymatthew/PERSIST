# persist/evolution/ewc_utils.py
import torch

def fisher_importance(model, data_loader) -> torch.Tensor:
    """
    Returns a placeholder for the diagonal Fisher information matrix.
    A real implementation would compute this based on the model and data.
    """
    # Placeholder: return a tensor of ones with the same shape as model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    importance = [torch.ones_like(p) for p in params]
    return torch.cat([i.view(-1) for i in importance])


def importance_mask(fisher_diag, quantile=0.7):
    """
    Creates a mask to identify important weights based on a Fisher diagonal.
    """
    if not isinstance(fisher_diag, torch.Tensor):
        fisher_diag = torch.tensor(fisher_diag)
    threshold = torch.quantile(fisher_diag, quantile)
    return fisher_diag >= threshold