# persist/evolution/evaluators/med_feature_sel.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def _kfold_indices(n_samples, n_splits=5, shuffle=False, random_state=None):
    """Simplified reimplementation of ``sklearn.model_selection.KFold``."""

    if n_splits <= 1:
        raise ValueError("n_splits must be at least 2")

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_indices = indices[start:stop]
        train_indices = np.concatenate((indices[:start], indices[stop:]))
        yield train_indices, test_indices
        current = stop


def _roc_auc_score(y_true, y_score):
    """Compute the ROC AUC for binary labels using the Mann-Whitney U statistic."""

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    pos_mask = y_true == 1
    neg_mask = y_true == 0

    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    if n_pos == 0 or n_neg == 0:
        # Degenerate case where AUC is undefined – match sklearn's behaviour by
        # returning 0.5 so downstream code still receives a float.
        return 0.5

    # Use average ranks to handle score ties deterministically.
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)

    auc = (ranks[pos_mask].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)

def demographic_parity(y_pred, sensitive_features):
    """
    Calculates the demographic parity difference.
    Assumes binary sensitive feature.
    """
    group_0 = y_pred[sensitive_features == 0]
    group_1 = y_pred[sensitive_features == 1]

    if len(group_0) == 0 or len(group_1) == 0:
        return 0.0

    return float(abs(group_0.mean() - group_1.mean()))


class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def crossval_auc_and_fairness(X, y, sensitive_features, n_splits=5):
    """
    Performs cross-validation to evaluate a feature set.
    """
    aucs, gaps = [], []

    for train_index, test_index in _kfold_indices(len(X), n_splits=n_splits, shuffle=True, random_state=42):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        sensitive_train, sensitive_test = sensitive_features[train_index], sensitive_features[test_index]

        model = LogisticRegression(X_train.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCELoss()

        # Training
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(torch.tensor(X_train, dtype=torch.float32))
            loss = criterion(outputs, torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
            loss.backward()
            optimizer.step()

        # Evaluation
        with torch.no_grad():
            y_pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy().flatten()
            aucs.append(_roc_auc_score(y_test, y_pred))
            gaps.append(demographic_parity(y_pred, sensitive_test))

    return np.mean(aucs), np.mean(gaps)


def eval_feature_mask(ind) -> dict:
    """
    Evaluates a feature mask for a medical dataset.
    """
    mask = ind["genes"]["mask"]      # e.g., numpy bool array

    # Create dummy data for the placeholder
    # In a real scenario, this data would be loaded from a file
    X = np.random.rand(100, len(mask))
    y = np.random.randint(0, 2, 100)
    sensitive_features = np.random.randint(0, 2, 100)

    # Apply the feature mask
    X_masked = X[:, mask]

    if X_masked.shape[1] == 0:
        return {"neg_auc": 0, "k": 0, "fair_gap": 1.0}

    auc, gap = crossval_auc_and_fairness(X_masked, y, sensitive_features)
    return {"neg_auc": -auc, "k": mask.sum(), "fair_gap": gap}