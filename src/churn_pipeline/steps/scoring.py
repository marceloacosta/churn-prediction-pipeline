"""
Scoring Utilities — Risk Tiers and Showing Your Work
=====================================================

When the model says "Customer X has a 0.73 probability of leaving," that's useful
for a data scientist. But a business user needs two things:

1. **A risk tier** — a simple label: "high", "medium", or "low". The business user
   doesn't care about 0.73 vs 0.71 — they care about "should I act NOW or later?"

2. **An explanation** — WHY does the model think this? SHAP (SHapley Additive
   exPlanations) gives us the answer. For each prediction, SHAP computes how much
   each feature pushed the probability up or down. It's like a courtroom verdict
   where the jury explains their reasoning: "month-to-month contract added +23%
   risk, low tenure added +15%, but having a partner reduced risk by -8%."

This module handles both:
- `assign_risk_tier()` — converts probability → human-friendly label
- `extract_top_reasons()` — picks the top 3 most influential features from SHAP
- `format_predictions()` — assembles the final client deliverable CSV
"""

from typing import List, Optional

import numpy as np
import pandas as pd


# Thresholds that define risk buckets
RISK_THRESHOLDS = {
    "high": 0.7,   # >= 0.7 → "this customer is very likely leaving — act NOW"
    "medium": 0.4,  # >= 0.4 and < 0.7 → "at risk — monitor closely"
    "low": 0.0,    # < 0.4 → "seems safe — no immediate action"
}


def assign_risk_tier(probability: float) -> str:
    """
    Convert a raw churn probability into a human-friendly risk label.

    Imagine a weather forecast: you don't say "there's a 0.73 probability of rain."
    You say "heavy rain likely." Risk tiers do the same translation for churn:

    - **high** (>= 0.7): "This customer is very likely leaving — intervene NOW"
    - **medium** (0.4 to 0.7): "At risk — monitor closely, consider reaching out"
    - **low** (< 0.4): "Seems safe — no immediate action needed"

    This function is deterministic: the same probability ALWAYS gets the same tier.
    It's also monotonic: a higher probability never gets a LOWER tier.

    Args:
        probability: Float in [0.0, 1.0] representing churn likelihood.

    Returns:
        One of "high", "medium", or "low".
    """
    if probability >= RISK_THRESHOLDS["high"]:
        return "high"
    elif probability >= RISK_THRESHOLDS["medium"]:
        return "medium"
    else:
        return "low"


def extract_top_reasons(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 3,
) -> List[str]:
    """
    From a customer's SHAP contributions, extract the top N most influential features.

    SHAP tells us how much each feature pushed the prediction up or down. A positive
    contribution means "this feature increased churn risk." A negative contribution
    means "this feature reduced churn risk."

    We sort by absolute value (magnitude of influence, regardless of direction)
    and take the top N. This gives the most impactful factors — whether they're
    pushing toward churn or away from it.

    Format: "feature_name (+0.23)" or "feature_name (-0.08)"

    Args:
        shap_values: Array of SHAP contributions for one customer.
            Shape: (n_features,). Each value is a signed float.
        feature_names: List of feature names matching the SHAP array order.
        top_n: Number of top reasons to extract (default: 3).

    Returns:
        List of exactly top_n formatted strings, sorted by absolute contribution
        (highest first). Example: ["contract_type (+0.23)", "tenure_months (+0.15)",
        "partner_status (-0.08)"]

    Raises:
        ValueError: If shap_values and feature_names have different lengths.
    """
    shap_values = np.asarray(shap_values, dtype=float)

    if len(shap_values) != len(feature_names):
        raise ValueError(
            f"shap_values length ({len(shap_values)}) must match "
            f"feature_names length ({len(feature_names)})"
        )

    # Sort by absolute contribution (most influential first)
    abs_contributions = np.abs(shap_values)
    top_indices = np.argsort(abs_contributions)[::-1][:top_n]

    reasons = []
    for idx in top_indices:
        value = shap_values[idx]
        name = feature_names[idx]
        sign = "+" if value >= 0 else ""
        reasons.append(f"{name} ({sign}{value:.2f})")

    return reasons


def format_predictions(
    customer_ids: List[str],
    probabilities: np.ndarray,
    shap_values: np.ndarray,
    feature_names: List[str],
    narratives: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Assemble the final client deliverable — the scored CSV.

    This is what the client actually receives: a spreadsheet telling them who's
    at risk, how much, why, and (optionally) a plain-English explanation.

    Output columns:
    - customer_id: Who is this about?
    - churn_probability: How likely are they to leave? (0.0 to 1.0)
    - risk_tier: Human-friendly label (high/medium/low)
    - top_3_reasons: SHAP-derived explanation of WHY
    - narrative_explanation: LLM-generated plain English (or "N/A")

    Args:
        customer_ids: List of customer identifiers.
        probabilities: Array of churn probabilities, shape (n_customers,).
        shap_values: SHAP matrix, shape (n_customers, n_features).
        feature_names: Feature names matching SHAP columns.
        narratives: Optional list of LLM-generated narratives.
            If None, fills with "N/A".

    Returns:
        DataFrame with the output CSV schema, ready to write to file.
    """
    probabilities = np.asarray(probabilities)
    shap_values = np.asarray(shap_values)
    n_customers = len(customer_ids)

    if narratives is None:
        narratives = ["N/A"] * n_customers

    rows = []
    for i in range(n_customers):
        prob = float(probabilities[i])
        tier = assign_risk_tier(prob)
        reasons = extract_top_reasons(shap_values[i], feature_names, top_n=3)
        reasons_str = "; ".join(reasons)

        rows.append({
            "customer_id": customer_ids[i],
            "churn_probability": round(prob, 4),
            "risk_tier": tier,
            "top_3_reasons": reasons_str,
            "narrative_explanation": narratives[i],
        })

    return pd.DataFrame(rows)
