"""
Property 2: Schema Validation Completeness
============================================

For ANY dataset that contains all Tier 1 fields (customer_id, tenure_months,
monthly_charges, total_charges, churn_label) with valid types, the validator
SHALL report is_valid=True regardless of which Tier 2 or Tier 3 fields are
present or absent.

Conversely, for ANY dataset missing at least one Tier 1 field, the validator
SHALL report is_valid=False.

This property ensures the validator's decision is determined entirely by Tier 1
completeness — optional fields never cause rejection.

Validates: Requirements 2.2, 2.3, 2.4

# Feature: churn-prediction-pipeline, Property 2: Schema Validation Completeness
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from churn_pipeline.data_contract import STANDARD_SCHEMA, Tier
from churn_pipeline.steps.validate_data import validate_dataset


# ---------------------------------------------------------------------------
# Helpers: build DataFrames with controlled tier coverage
# ---------------------------------------------------------------------------

# Tier 1 field names (the non-negotiables)
TIER1_FIELDS = [name for name, spec in STANDARD_SCHEMA.items() if spec.tier == Tier.REQUIRED]

# Tier 2 field names (engagement)
TIER2_FIELDS = [name for name, spec in STANDARD_SCHEMA.items() if spec.tier == Tier.ENGAGEMENT]

# Tier 3 field names (demographics)
TIER3_FIELDS = [name for name, spec in STANDARD_SCHEMA.items() if spec.tier == Tier.DEMOGRAPHIC]


def _build_tier1_data(n_rows: int) -> dict:
    """Generate valid data for all Tier 1 fields."""
    return {
        "customer_id": [f"CUST_{i:04d}" for i in range(n_rows)],
        "tenure_months": np.random.randint(1, 72, size=n_rows).tolist(),
        "monthly_charges": np.random.uniform(18.0, 118.0, size=n_rows).tolist(),
        "total_charges": np.random.uniform(100.0, 8000.0, size=n_rows).tolist(),
        "churn_label": np.random.randint(0, 2, size=n_rows).tolist(),
    }


def _build_tier2_data(n_rows: int, fields: list) -> dict:
    """Generate valid data for specified Tier 2 fields."""
    data = {}
    for field_name in fields:
        if field_name == "contract_type":
            data[field_name] = np.random.choice(
                ["month-to-month", "one_year", "two_year"], size=n_rows
            ).tolist()
        elif field_name == "payment_method":
            data[field_name] = np.random.choice(
                ["credit_card", "bank_transfer", "electronic_check"], size=n_rows
            ).tolist()
        elif field_name == "support_tickets":
            data[field_name] = np.random.randint(0, 10, size=n_rows).tolist()
    return data


def _build_tier3_data(n_rows: int, fields: list) -> dict:
    """Generate valid data for specified Tier 3 fields."""
    data = {}
    for field_name in fields:
        if field_name == "gender":
            data[field_name] = np.random.choice(["Male", "Female"], size=n_rows).tolist()
        elif field_name == "age_bucket":
            data[field_name] = np.random.choice(
                ["18-25", "26-35", "36-45", "46-55", "56+"], size=n_rows
            ).tolist()
        elif field_name == "partner_status":
            data[field_name] = np.random.choice(["Yes", "No"], size=n_rows).tolist()
    return data


# ---------------------------------------------------------------------------
# Property test: ALL Tier 1 present + random Tier 2/3 → always valid
# ---------------------------------------------------------------------------

@pytest.mark.property
@given(
    n_rows=st.integers(min_value=1, max_value=100),
    tier2_subset=st.lists(st.sampled_from(TIER2_FIELDS), unique=True),
    tier3_subset=st.lists(st.sampled_from(TIER3_FIELDS), unique=True),
)
@settings(max_examples=200)
def test_all_tier1_present_always_valid(
    n_rows: int,
    tier2_subset: list,
    tier3_subset: list,
) -> None:
    """
    With ALL Tier 1 fields present (valid types), the dataset must be valid
    regardless of which Tier 2/3 fields are included or absent.
    """
    # Build DataFrame with all Tier 1 + random subset of Tier 2/3
    data = _build_tier1_data(n_rows)
    data.update(_build_tier2_data(n_rows, tier2_subset))
    data.update(_build_tier3_data(n_rows, tier3_subset))

    df = pd.DataFrame(data)
    result = validate_dataset(df)

    # The verdict MUST be valid
    assert result.is_valid is True, (
        f"Expected is_valid=True with all Tier 1 fields present. "
        f"tier2={tier2_subset}, tier3={tier3_subset}, errors={result.errors}"
    )

    # All Tier 1 fields must be in the "present" list
    assert set(result.tier1_present) == set(TIER1_FIELDS)
    assert result.tier1_missing == []

    # Tier 2/3 present/missing must match what we included
    assert set(result.tier2_present) == set(tier2_subset)
    assert set(result.tier2_missing) == set(TIER2_FIELDS) - set(tier2_subset)
    assert set(result.tier3_present) == set(tier3_subset)
    assert set(result.tier3_missing) == set(TIER3_FIELDS) - set(tier3_subset)


# ---------------------------------------------------------------------------
# Property test: missing at least one Tier 1 field → always invalid
# ---------------------------------------------------------------------------

@pytest.mark.property
@given(
    n_rows=st.integers(min_value=1, max_value=100),
    tier1_to_drop=st.lists(
        st.sampled_from(TIER1_FIELDS), min_size=1, unique=True
    ),
    tier2_subset=st.lists(st.sampled_from(TIER2_FIELDS), unique=True),
    tier3_subset=st.lists(st.sampled_from(TIER3_FIELDS), unique=True),
)
@settings(max_examples=200)
def test_missing_tier1_always_invalid(
    n_rows: int,
    tier1_to_drop: list,
    tier2_subset: list,
    tier3_subset: list,
) -> None:
    """
    With at least one Tier 1 field missing, the dataset must be invalid
    regardless of what Tier 2/3 fields are present.
    """
    # Build full DataFrame, then drop some Tier 1 fields
    data = _build_tier1_data(n_rows)
    data.update(_build_tier2_data(n_rows, tier2_subset))
    data.update(_build_tier3_data(n_rows, tier3_subset))

    df = pd.DataFrame(data)
    df = df.drop(columns=tier1_to_drop)

    result = validate_dataset(df)

    # The verdict MUST be invalid
    assert result.is_valid is False, (
        f"Expected is_valid=False when Tier 1 fields {tier1_to_drop} are missing. "
        f"Got is_valid=True."
    )

    # The dropped fields must appear in tier1_missing
    assert set(tier1_to_drop).issubset(set(result.tier1_missing)), (
        f"Dropped {tier1_to_drop} but tier1_missing={result.tier1_missing}"
    )
