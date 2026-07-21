"""
Drift Monitoring — The Smoke Detector for When the World Changes
=================================================================

When we trained the model, customers looked a certain way — maybe average tenure
was 24 months and most paid $60/month. If next month's batch has average tenure
of 6 months and everyone pays $120/month, the model is seeing a world it wasn't
trained for. Its predictions become unreliable because the patterns it learned
may no longer apply.

PSI (Population Stability Index) is our smoke detector. It answers: "Does today's
data look like the data the model was trained on?"

**How PSI works (step by step):**
1. During training, we recorded how customers were distributed across each feature.
   For example: 30% had tenure 0-12, 40% had 12-36, 30% had 36+.
2. When new scoring data comes in, we count the same buckets.
   Maybe now: 60% have tenure 0-12 (a big shift!).
3. PSI measures the divergence between these two distributions:
   PSI = Σ (new_% - old_%) × ln(new_% / old_%)
4. Interpretation:
   - PSI < 0.1  → stable (the world looks the same)
   - PSI 0.1-0.2 → moderate shift (something changed, watch closely)
   - PSI > 0.2  → significant change (model may be unreliable, consider retraining)

A special case: if you compute PSI of a distribution against itself, the answer
is always 0.0 (no difference). This is how we verify our implementation is correct.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class DriftReport:
    """
    The smoke detector's report — what did it find?

    Attributes:
        run_date: When this check was performed.
        features_checked: How many features were tested for drift.
        features_drifted: Names of features that exceeded the PSI threshold.
        psi_scores: Per-feature PSI values (feature_name → PSI).
        alert_triggered: True if ANY feature exceeded the threshold.
    """

    run_date: str
    features_checked: int
    features_drifted: List[str] = field(default_factory=list)
    psi_scores: Dict[str, float] = field(default_factory=dict)
    alert_triggered: bool = False


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> float:
    """
    Compute Population Stability Index between two distributions.

    Step by step:
    1. Divide the reference data into N equal-width buckets based on its range.
    2. Count what percentage of reference values fall in each bucket.
    3. Count what percentage of current values fall in each bucket.
    4. For each bucket: (current_% - reference_%) × ln(current_% / reference_%)
    5. Sum across all buckets → that's PSI.

    Result interpretation:
    - 0.0: identical distributions (comparing something to itself)
    - < 0.1: negligible drift (safe)
    - 0.1-0.2: moderate drift (watch)
    - > 0.2: significant drift (alert, consider retraining)

    Args:
        reference: The training-time distribution (1D array of floats).
        current: The new data distribution (1D array of floats).
        bins: Number of buckets to divide the data into (default: 10).

    Returns:
        PSI value (non-negative float). Zero means identical distributions.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    # Create bin edges from the reference distribution
    # Use the reference range so bins are consistent across comparisons
    min_val = reference.min()
    max_val = reference.max()

    # Handle edge case: constant feature (all same value)
    if min_val == max_val:
        # If current is also constant and same value, PSI = 0
        if current.min() == current.max() == min_val:
            return 0.0
        # Otherwise there's drift, but we can't bucket properly
        # Return a high PSI to signal the change
        return 0.25

    # Create equal-width bins spanning the reference range
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    # Extend edges slightly to capture values at the boundaries
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Count proportions in each bucket
    ref_counts = np.histogram(reference, bins=bin_edges)[0]
    cur_counts = np.histogram(current, bins=bin_edges)[0]

    # Convert to proportions
    ref_proportions = ref_counts / len(reference)
    cur_proportions = cur_counts / len(current)

    # Replace zeros with a small epsilon to avoid division by zero and log(0)
    epsilon = 1e-6
    ref_proportions = np.maximum(ref_proportions, epsilon)
    cur_proportions = np.maximum(cur_proportions, epsilon)

    # PSI formula: Σ (current - reference) × ln(current / reference)
    psi = np.sum(
        (cur_proportions - ref_proportions) * np.log(cur_proportions / ref_proportions)
    )

    return float(psi)


def check_drift(
    training_stats: Dict[str, np.ndarray],
    current_data: pd.DataFrame,
    threshold: float = 0.2,
) -> DriftReport:
    """
    Compare current scoring batch against training distributions.

    For each feature that was tracked during training, compute PSI against
    the current batch. If any feature exceeds the threshold, trigger an alert.

    Args:
        training_stats: Dict mapping feature names to their training-time
            distributions (1D numpy arrays of values seen during training).
        current_data: DataFrame of current scoring data (after feature engineering
            or at least with standardized column names).
        threshold: PSI value above which drift is considered significant
            (default: 0.2, meaning "the world changed significantly").

    Returns:
        DriftReport with per-feature PSI scores and alert status.
    """
    psi_scores: Dict[str, float] = {}
    features_drifted: List[str] = []

    for feature_name, reference_values in training_stats.items():
        if feature_name not in current_data.columns:
            continue

        current_values = current_data[feature_name].dropna().values

        # Skip if either distribution is too small for meaningful comparison
        if len(current_values) < 10 or len(reference_values) < 10:
            continue

        psi = compute_psi(reference_values, current_values)
        psi_scores[feature_name] = psi

        if psi > threshold:
            features_drifted.append(feature_name)

    return DriftReport(
        run_date="",  # Caller fills this in
        features_checked=len(psi_scores),
        features_drifted=features_drifted,
        psi_scores=psi_scores,
        alert_triggered=len(features_drifted) > 0,
    )
