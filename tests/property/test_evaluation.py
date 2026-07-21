"""
Property 6: Evaluation Gate Determinism
=========================================

For ANY set of model metrics (auc_roc, f1_score, precision, recall — all floats
in [0, 1]), the evaluation gate pass/fail decision SHALL depend solely on whether
auc_roc >= 0.70, independent of all other metric values.

A model with AUC 0.71, F1 0.10, precision 0.05, recall 0.90 must still pass.
A model with AUC 0.69, F1 0.99, precision 0.99, recall 0.99 must still fail.
Only AUC controls the gate.

Validates: Requirements 5.1, 5.2

# Feature: churn-prediction-pipeline, Property 6: Evaluation Gate Determinism
"""

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from churn_pipeline.steps.evaluation import evaluate_model


@st.composite
def evaluation_data_passing(draw):
    """
    Generate test data where AUC will be >= 0.70.
    We achieve this by creating labels and predictions with strong positive correlation.
    """
    n = draw(st.integers(min_value=50, max_value=200))
    # Create labels with some imbalance
    n_pos = draw(st.integers(min_value=max(5, n // 10), max_value=n // 2))
    y_true = np.array([0] * (n - n_pos) + [1] * n_pos)
    np.random.shuffle(y_true)

    # Create predictions that correlate well with true labels (high AUC)
    y_pred = np.where(
        y_true == 1,
        np.random.uniform(0.6, 1.0, size=n),  # Positives get high probs
        np.random.uniform(0.0, 0.4, size=n),  # Negatives get low probs
    )

    return y_true, y_pred


@st.composite
def evaluation_data_failing(draw):
    """
    Generate test data where AUC will be < 0.70.
    We achieve this by making predictions nearly random (weak correlation).
    """
    n = draw(st.integers(min_value=50, max_value=200))
    n_pos = draw(st.integers(min_value=max(5, n // 10), max_value=n // 2))
    y_true = np.array([0] * (n - n_pos) + [1] * n_pos)
    np.random.shuffle(y_true)

    # Create predictions that are mostly random (low AUC)
    y_pred = np.random.uniform(0.3, 0.7, size=n)  # All predictions near 0.5

    return y_true, y_pred


@pytest.mark.property
@given(data=evaluation_data_passing())
@settings(max_examples=200, deadline=None)
def test_high_auc_always_passes(data) -> None:
    """When AUC >= 0.70, the gate must pass regardless of other metrics."""
    y_true, y_pred = data

    result = evaluate_model(y_true, y_pred, min_auc_threshold=0.70)

    if result.auc_roc >= 0.70:
        assert result.passed is True, (
            f"AUC={result.auc_roc} >= 0.70 but gate failed. "
            f"F1={result.f1_score}, precision={result.precision}, recall={result.recall}"
        )


@pytest.mark.property
@given(data=evaluation_data_failing())
@settings(max_examples=200, deadline=None)
def test_low_auc_always_fails(data) -> None:
    """When AUC < 0.70, the gate must fail regardless of other metrics."""
    y_true, y_pred = data

    result = evaluate_model(y_true, y_pred, min_auc_threshold=0.70)

    if result.auc_roc < 0.70:
        assert result.passed is False, (
            f"AUC={result.auc_roc} < 0.70 but gate passed. "
            f"F1={result.f1_score}, precision={result.precision}, recall={result.recall}"
        )
