import importlib
import sys

import numpy as np
import pytest
from evolution.evaluators.med_feature_sel import crossval_auc_and_fairness
from evolution.evaluators.rl_metaeval import train_and_measure


def test_med_feature_sel_import_without_sklearn(monkeypatch):
    """The evaluator should remain importable when scikit-learn is missing."""

    module_name = "evolution.evaluators.med_feature_sel"
    sys.modules.pop(module_name, None)

    original_import = __import__

    def fail_on_sklearn(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ModuleNotFoundError("No module named 'sklearn'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fail_on_sklearn)

    module = importlib.import_module(module_name)
    assert hasattr(module, "crossval_auc_and_fairness")

def test_crossval_auc_and_fairness():
    """
    Tests the crossval_auc_and_fairness function with synthetic data.
    """
    X = np.random.rand(50, 10)
    y = np.random.randint(0, 2, 50)
    sensitive_features = np.random.randint(0, 2, 50)

    auc, gap = crossval_auc_and_fairness(X, y, sensitive_features, n_splits=2)

    assert isinstance(auc, float)
    assert 0.0 <= auc <= 1.0
    assert isinstance(gap, float)
    assert 0.0 <= gap <= 1.0

def test_train_and_measure():
    """
    Tests the train_and_measure function for the RL evaluator.
    """
    cfg = {
        "net_width": 16,
        "net_depth": 2,
        "lr": 1e-3,
        "gamma": 0.99
    }

    avg_return, violations, latency = train_and_measure(cfg, max_episodes=2)

    assert isinstance(avg_return, float)
    assert isinstance(violations, float)
    assert isinstance(latency, float)
    assert latency > 0