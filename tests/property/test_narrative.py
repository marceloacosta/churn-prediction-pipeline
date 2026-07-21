"""
Property 12: Narrative Prompt Completeness
============================================

For ANY batch of scored customers (each with customer_id, churn_probability,
risk_tier, and SHAP features), the prompt construction function SHALL produce
a prompt string that contains every customer's ID, probability, tier, and all
SHAP feature names with contribution values.

If the prompt is missing information, the LLM can't write accurate narratives.
This ensures we always give Claude everything it needs.

Validates: Requirements 7.3

# Feature: churn-prediction-pipeline, Property 12: Narrative Prompt Completeness
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from churn_pipeline.llm.narrative_generator import (
    NarrativeRequest,
    build_narrative_prompt,
)


# Strategy for feature names
_feature_name = st.text(
    alphabet=st.characters(whitelist_categories=("L",), whitelist_characters="_"),
    min_size=3,
    max_size=15,
)

# Strategy for SHAP feature dicts
_shap_feature = st.fixed_dictionaries({
    "feature": _feature_name,
    "contribution": st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
})

# Strategy for NarrativeRequest
_narrative_request = st.builds(
    NarrativeRequest,
    customer_id=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
        min_size=3,
        max_size=15,
    ).filter(lambda s: s[0].isalpha()),
    churn_probability=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    risk_tier=st.sampled_from(["high", "medium", "low"]),
    top_shap_features=st.lists(_shap_feature, min_size=1, max_size=5),
)


@pytest.mark.property
@given(batch=st.lists(_narrative_request, min_size=1, max_size=10))
@settings(max_examples=200, deadline=None)
def test_narrative_prompt_contains_all_customer_info(
    batch: list,
) -> None:
    """
    The prompt must contain every customer's ID, probability, tier,
    and all SHAP feature names.
    """
    prompt = build_narrative_prompt(batch)

    for req in batch:
        # Customer ID must appear
        assert req.customer_id in prompt, (
            f"Customer ID '{req.customer_id}' not found in prompt"
        )

        # Probability must appear (formatted to 2 decimal places)
        prob_str = f"{req.churn_probability:.2f}"
        assert prob_str in prompt, (
            f"Probability '{prob_str}' for customer '{req.customer_id}' not in prompt"
        )

        # Risk tier must appear
        assert req.risk_tier in prompt, (
            f"Risk tier '{req.risk_tier}' for customer '{req.customer_id}' not in prompt"
        )

        # Each SHAP feature name must appear
        for feat in req.top_shap_features:
            feature_name = feat["feature"]
            assert feature_name in prompt, (
                f"SHAP feature '{feature_name}' for customer "
                f"'{req.customer_id}' not in prompt"
            )
