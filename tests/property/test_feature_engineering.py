"""
Property 3: Feature Matrix Integrity
Property 11: Feature Engineering Idempotence (Scoring Mode)
============================================================

Property 3:
For ANY schema-conformant input DataFrame with N rows (possibly containing null
values and mixed types), the feature engineering module SHALL produce a feature
matrix with exactly N rows, all numeric (float64) columns, and zero null values.

Property 11:
For ANY input DataFrame and fixed feature artifacts, applying feature engineering
in scoring mode (fit=False) twice with the same artifacts SHALL produce bit-for-bit
identical output both times.

Validates: Requirements 3.1, 3.2, 3.3, 3.5

# Feature: churn-prediction-pipeline, Property 3: Feature Matrix Integrity
# Feature: churn-prediction-pipeline, Property 11: Feature Engineering Idempotence
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from churn_pipeline.steps.feature_engineering import engineer_features, FeatureArtifacts


# ---------------------------------------------------------------------------
# Hypothesis strategies for generating schema-conformant DataFrames
# ---------------------------------------------------------------------------

# Generate realistic numeric values with occasional nulls
_tenure = st.one_of(
    st.integers(min_value=1, max_value=72),
    st.none(),
)
_charges = st.one_of(
    st.floats(min_value=10.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    st.none(),
)
_total = st.one_of(
    st.floats(min_value=10.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    st.none(),
)
_tickets = st.one_of(
    st.integers(min_value=0, max_value=20),
    st.none(),
)
_contract = st.one_of(
    st.sampled_from(["month-to-month", "one_year", "two_year"]),
    st.none(),
)
_payment = st.one_of(
    st.sampled_from(["credit_card", "bank_transfer", "electronic_check"]),
    st.none(),
)
_churn = st.integers(min_value=0, max_value=1)


@st.composite
def schema_conformant_dataframe(draw):
    """Generate a random DataFrame with Tier 1 fields and optional Tier 2/3."""
    n_rows = draw(st.integers(min_value=2, max_value=50))

    # Always include Tier 1 fields (values may have nulls, which imputation handles)
    data = {
        "customer_id": [f"C{i:04d}" for i in range(n_rows)],
        "tenure_months": [draw(_tenure) for _ in range(n_rows)],
        "monthly_charges": [draw(_charges) for _ in range(n_rows)],
        "total_charges": [draw(_total) for _ in range(n_rows)],
        "churn_label": [draw(_churn) for _ in range(n_rows)],
    }

    # Optionally include Tier 2 fields
    if draw(st.booleans()):
        data["contract_type"] = [draw(_contract) for _ in range(n_rows)]
    if draw(st.booleans()):
        data["payment_method"] = [draw(_payment) for _ in range(n_rows)]
    if draw(st.booleans()):
        data["support_tickets"] = [draw(_tickets) for _ in range(n_rows)]

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Property 3: Feature Matrix Integrity
# ---------------------------------------------------------------------------

@pytest.mark.property
@given(df=schema_conformant_dataframe())
@settings(max_examples=200, deadline=None)
def test_feature_matrix_integrity(df: pd.DataFrame) -> None:
    """
    For any schema-conformant DataFrame with N rows:
    - Output has exactly N rows
    - Output dtype is float64
    - Output contains zero NaN values
    """
    n_rows = len(df)

    matrix, artifacts = engineer_features(df, fit=True)

    # Guarantee 1: Same row count (no rows lost or duplicated)
    assert matrix.shape[0] == n_rows, (
        f"Row count mismatch: input had {n_rows} rows, output has {matrix.shape[0]}"
    )

    # Guarantee 2: All float64
    assert matrix.dtype == np.float64, (
        f"Expected float64, got {matrix.dtype}"
    )

    # Guarantee 3: Zero NaN values
    assert not np.isnan(matrix).any(), (
        f"Found {np.isnan(matrix).sum()} NaN values in output matrix"
    )

    # Bonus: feature names list matches column count
    assert matrix.shape[1] == len(artifacts.feature_names), (
        f"Matrix has {matrix.shape[1]} columns but artifacts list {len(artifacts.feature_names)} names"
    )


# ---------------------------------------------------------------------------
# Property 11: Feature Engineering Idempotence (Scoring Mode)
# ---------------------------------------------------------------------------

@pytest.mark.property
@given(df=schema_conformant_dataframe())
@settings(max_examples=200, deadline=None)
def test_feature_engineering_idempotence(df: pd.DataFrame) -> None:
    """
    For any input DataFrame and fixed artifacts, applying feature engineering
    in scoring mode twice produces bit-for-bit identical output.
    """
    # First, fit to get artifacts
    _, artifacts = engineer_features(df, fit=True)

    # Run scoring mode twice with same artifacts
    result_1, _ = engineer_features(df, fit=False, artifacts=artifacts)
    result_2, _ = engineer_features(df, fit=False, artifacts=artifacts)

    # Must be bit-for-bit identical
    assert np.array_equal(result_1, result_2), (
        f"Scoring mode produced different results on two runs. "
        f"Max difference: {np.abs(result_1 - result_2).max()}"
    )
