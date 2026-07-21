"""
Property 7: Risk Tier Assignment Consistency and Monotonicity
Property 8: SHAP Explanation Completeness
==========================================================

Property 7:
For ANY churn probability p in [0.0, 1.0], the risk tier assignment SHALL be
deterministic (same p always gives same tier). Additionally, for ANY two
probabilities p1 > p2, the risk tier of p1 SHALL be >= the risk tier of p2.

Property 8:
For ANY scored customer with SHAP values across F features (where F >= 3),
the top-3 extraction function SHALL return exactly 3 reasons, each referencing
a feature name that exists in the model's feature list, sorted by absolute
contribution (highest first).

Validates: Requirements 6.3, 6.4

# Feature: churn-prediction-pipeline, Property 7: Risk Tier Assignment Consistency and Monotonicity
# Feature: churn-prediction-pipeline, Property 8: SHAP Explanation Completeness
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from churn_pipeline.steps.scoring import assign_risk_tier, extract_top_reasons


# Map tiers to numeric ranks for monotonicity comparison
_TIER_RANK = {"low": 0, "medium": 1, "high": 2}


# ---------------------------------------------------------------------------
# Property 7: Risk Tier Assignment Consistency and Monotonicity
# ---------------------------------------------------------------------------

@pytest.mark.property
@given(probability=st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=200)
def test_risk_tier_deterministic(probability: float) -> None:
    """Same probability always produces the same tier (no randomness)."""
    tier_1 = assign_risk_tier(probability)
    tier_2 = assign_risk_tier(probability)

    assert tier_1 == tier_2, (
        f"Non-deterministic: {probability} → '{tier_1}' then '{tier_2}'"
    )
    assert tier_1 in ("high", "medium", "low"), (
        f"Invalid tier '{tier_1}' for probability {probability}"
    )


@pytest.mark.property
@given(
    p1=st.floats(min_value=0.0, max_value=1.0),
    p2=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=200)
def test_risk_tier_monotonic(p1: float, p2: float) -> None:
    """Higher probability never gets a LOWER risk tier."""
    tier_1 = assign_risk_tier(p1)
    tier_2 = assign_risk_tier(p2)

    if p1 > p2:
        assert _TIER_RANK[tier_1] >= _TIER_RANK[tier_2], (
            f"Monotonicity violated: p1={p1} ('{tier_1}') > p2={p2} ('{tier_2}') "
            f"but tier rank {_TIER_RANK[tier_1]} < {_TIER_RANK[tier_2]}"
        )
    elif p2 > p1:
        assert _TIER_RANK[tier_2] >= _TIER_RANK[tier_1], (
            f"Monotonicity violated: p2={p2} ('{tier_2}') > p1={p1} ('{tier_1}') "
            f"but tier rank {_TIER_RANK[tier_2]} < {_TIER_RANK[tier_1]}"
        )


# ---------------------------------------------------------------------------
# Property 8: SHAP Explanation Completeness
# ---------------------------------------------------------------------------

# Strategy for feature name lists (3 to 20 features)
_feature_names = st.lists(
    st.text(
        alphabet=st.characters(whitelist_categories=("L",), whitelist_characters="_"),
        min_size=2,
        max_size=15,
    ),
    min_size=3,
    max_size=20,
    unique=True,
)


@st.composite
def shap_with_features(draw):
    """Generate matching SHAP values and feature names."""
    features = draw(_feature_names)
    n_features = len(features)
    # SHAP values: random floats, some positive, some negative
    shap_values = draw(
        st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=n_features,
            max_size=n_features,
        )
    )
    return np.array(shap_values), features


@pytest.mark.property
@given(data=shap_with_features())
@settings(max_examples=200)
def test_shap_extraction_completeness(data) -> None:
    """
    Top-3 extraction always returns exactly 3 reasons, each referencing
    a real feature name, sorted by absolute contribution.
    """
    shap_values, feature_names = data

    reasons = extract_top_reasons(shap_values, feature_names, top_n=3)

    # Must return exactly 3 reasons
    assert len(reasons) == 3, (
        f"Expected exactly 3 reasons, got {len(reasons)}: {reasons}"
    )

    # Each reason must reference a feature from the list
    for reason in reasons:
        # Format is "feature_name (+0.23)" or "feature_name (-0.08)"
        feature_part = reason.rsplit(" (", 1)[0]
        assert feature_part in feature_names, (
            f"Reason '{reason}' references unknown feature '{feature_part}'. "
            f"Known features: {feature_names}"
        )

    # Reasons must be sorted by absolute contribution (highest first)
    contributions = []
    for reason in reasons:
        # Extract the numeric part from "feature_name (+0.23)"
        value_str = reason.rsplit("(", 1)[1].rstrip(")")
        contributions.append(abs(float(value_str)))

    for i in range(len(contributions) - 1):
        assert contributions[i] >= contributions[i + 1], (
            f"Reasons not sorted by absolute contribution: "
            f"{contributions[i]:.4f} < {contributions[i+1]:.4f}. "
            f"Full reasons: {reasons}"
        )
