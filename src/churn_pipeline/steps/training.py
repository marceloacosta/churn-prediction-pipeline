"""
Training Utilities — Splitting Data Fairly and Handling Imbalance
==================================================================

Before you can teach a model, you need to divide your data into groups:

1. **Training set (70%):** The textbook — the model reads this to learn patterns.
2. **Validation set (15%):** The practice exam — used during training to check
   if the model is memorizing vs. actually learning.
3. **Test set (15%):** The final exam — never seen during training. This is the
   honest measure of how good the model really is.

But there's a catch: if 20% of customers churned overall, you need each group
to have ~20% churners. Otherwise the model might train on a group where 30%
churned and test on a group where 10% churned — the "world" looks different
in each group, and your performance estimate is unreliable.

This is called **stratified splitting** — preserving the class balance in every split.

The second problem: **class imbalance**. If only 15% of customers churn, a lazy
model can just predict "won't churn" for everyone and be right 85% of the time.
That's useless. `scale_pos_weight` tells the model: "Pay extra attention to the
rare churners — their mistakes cost more." It's a multiplier equal to
count(non-churners) / count(churners).
"""

from typing import Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def create_stratified_splits(
    features: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Split data into train/validation/test sets while preserving class proportions.

    Imagine a bag of marbles: 80 blue and 20 red. If you grab a handful randomly,
    you might get 15 blue and 0 red — that handful doesn't represent the bag.
    Stratified splitting ensures every handful has roughly the same blue/red ratio
    as the original bag.

    Args:
        features: Feature matrix of shape (n_samples, n_features).
        labels: Binary label array of shape (n_samples,). Values: 0 or 1.
        train_ratio: Fraction for training (default 0.70).
        val_ratio: Fraction for validation (default 0.15).
        test_ratio: Fraction for testing (default 0.15).
        random_state: Seed for reproducibility.

    Returns:
        Three tuples: (X_train, y_train), (X_val, y_val), (X_test, y_test)

    Raises:
        ValueError: If ratios don't sum to ~1.0 or labels aren't binary.
    """
    # Sanity checks
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"Ratios must sum to 1.0, got {total:.3f} "
            f"({train_ratio} + {val_ratio} + {test_ratio})"
        )

    if len(features) != len(labels):
        raise ValueError(
            f"features and labels must have same length, "
            f"got {len(features)} and {len(labels)}"
        )

    # First split: separate test set from the rest
    # test_ratio relative to the whole dataset
    X_rest, X_test, y_rest, y_test = train_test_split(
        features,
        labels,
        test_size=test_ratio,
        stratify=labels,
        random_state=random_state,
    )

    # Second split: separate validation from training
    # val_ratio relative to the remaining data (not the whole dataset)
    val_relative = val_ratio / (train_ratio + val_ratio)

    X_train, X_val, y_train, y_val = train_test_split(
        X_rest,
        y_rest,
        test_size=val_relative,
        stratify=y_rest,
        random_state=random_state,
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def compute_scale_pos_weight(labels: np.ndarray) -> float:
    """
    Compute how much extra attention the model should pay to the minority class.

    When few customers churn (say 15%), the model can be lazy: predict "won't churn"
    for everyone and be right 85% of the time. This weight says: "Each churner
    counts as N non-churners" — making the model work harder to get churners right.

    Formula: count(non-churners) / count(churners)

    Example:
        5950 stayed / 1050 left = 5.67
        → "Each churner counts as 5.67 customers in the loss function"

    Args:
        labels: Binary array where 1 = churned (positive class), 0 = stayed.

    Returns:
        The scale_pos_weight value. Returns 1.0 if classes are balanced
        (minority >= 30% of dataset).
    """
    labels = np.asarray(labels)
    n_positive = int(labels.sum())
    n_negative = len(labels) - n_positive

    if n_positive == 0:
        raise ValueError("Cannot compute scale_pos_weight: no positive samples (churners)")

    # Only apply weight if minority class < 30%
    minority_ratio = n_positive / len(labels)
    if minority_ratio >= 0.30:
        return 1.0

    return n_negative / n_positive


# ---------------------------------------------------------------------------
# Hyperparameter ranges for XGBoost tuning
# ---------------------------------------------------------------------------
# These are the "oven settings" the tuner tries different combinations of.
# Each one controls a different aspect of how the model learns.

HYPERPARAMETER_RANGES: Dict[str, Dict] = {
    "max_depth": {
        "type": "integer",
        "min": 3,
        "max": 10,
        "description": (
            "How many yes/no questions each tree can ask. "
            "Deeper = more complex patterns, but too deep = memorization."
        ),
    },
    "eta": {
        "type": "continuous",
        "min": 0.01,
        "max": 0.3,
        "description": (
            "Learning rate — how much each new tree corrects previous ones. "
            "Low = cautious (needs more trees), high = aggressive (may overreact)."
        ),
    },
    "subsample": {
        "type": "continuous",
        "min": 0.5,
        "max": 1.0,
        "description": (
            "What fraction of customers each tree sees. "
            "Using less (e.g. 70%) prevents memorization."
        ),
    },
    "colsample_bytree": {
        "type": "continuous",
        "min": 0.5,
        "max": 1.0,
        "description": (
            "What fraction of features each tree considers. "
            "Prevents any single feature from dominating."
        ),
    },
    "min_child_weight": {
        "type": "integer",
        "min": 1,
        "max": 10,
        "description": (
            "How many customers a tree needs before making a rule. "
            "Higher = more conservative (won't make rules based on 1-2 customers)."
        ),
    },
}
