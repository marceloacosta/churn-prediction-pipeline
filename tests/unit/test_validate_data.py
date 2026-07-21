"""
Unit Tests: Schema Validation Edge Cases
==========================================

These test specific, known scenarios that the property tests might not hit —
boundary conditions and degenerate inputs that exercise the validator's
error handling paths.

Validates: Requirements 2.2, 2.3, 2.4

# Feature: churn-prediction-pipeline, Task 2.3: Schema validation edge cases
"""

import pandas as pd
import pytest

from churn_pipeline.data_contract import STANDARD_SCHEMA, Tier
from churn_pipeline.steps.validate_data import validate_dataset


TIER1_FIELDS = [name for name, spec in STANDARD_SCHEMA.items() if spec.tier == Tier.REQUIRED]
TIER2_FIELDS = [name for name, spec in STANDARD_SCHEMA.items() if spec.tier == Tier.ENGAGEMENT]
TIER3_FIELDS = [name for name, spec in STANDARD_SCHEMA.items() if spec.tier == Tier.DEMOGRAPHIC]


class TestEmptyDataFrame:
    """An empty DataFrame should always be invalid — no data means no predictions."""

    def test_empty_dataframe_is_invalid(self):
        df = pd.DataFrame()
        result = validate_dataset(df)

        assert result.is_valid is False

    def test_empty_dataframe_reports_all_tier1_missing(self):
        df = pd.DataFrame()
        result = validate_dataset(df)

        assert set(result.tier1_missing) == set(TIER1_FIELDS)
        assert result.tier1_present == []

    def test_empty_dataframe_has_error_message(self):
        df = pd.DataFrame()
        result = validate_dataset(df)

        assert len(result.errors) > 0
        assert "empty" in result.errors[0].lower()

    def test_dataframe_with_columns_but_zero_rows(self):
        """Columns exist but no data rows — still invalid."""
        df = pd.DataFrame(columns=TIER1_FIELDS)
        result = validate_dataset(df)

        assert result.is_valid is False


class TestTypeMismatches:
    """Wrong types in Tier 1 fields should produce error messages."""

    def _make_valid_df(self) -> pd.DataFrame:
        """Helper to build a minimal valid DataFrame."""
        return pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "tenure_months": [12, 24, 36],
            "monthly_charges": [50.0, 75.0, 100.0],
            "total_charges": [600.0, 1800.0, 3600.0],
            "churn_label": [0, 1, 0],
        })

    def test_string_in_numeric_field_produces_error(self):
        """tenure_months contains a non-numeric string — error reported."""
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "tenure_months": ["twelve", "twenty-four", "abc"],
            "monthly_charges": [50.0, 75.0, 100.0],
            "total_charges": [600.0, 1800.0, 3600.0],
            "churn_label": [0, 1, 0],
        })
        result = validate_dataset(df)

        # Still valid (field IS present), but errors should be reported
        assert result.is_valid is True
        assert any("tenure_months" in err for err in result.errors)

    def test_mixed_numeric_and_string_reports_count(self):
        """total_charges has some convertible and some non-convertible values."""
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "tenure_months": [12, 24, 36],
            "monthly_charges": [50.0, 75.0, 100.0],
            "total_charges": ["600.0", " ", "3600.0"],  # " " is IBM Telco's quirk
            "churn_label": [0, 1, 0],
        })
        result = validate_dataset(df)

        assert result.is_valid is True
        # The " " value can't be converted to float
        assert any("total_charges" in err for err in result.errors)

    def test_valid_numeric_types_no_errors(self):
        """Properly typed fields should produce zero errors."""
        df = self._make_valid_df()
        result = validate_dataset(df)

        assert result.is_valid is True
        assert result.errors == []


class TestTier1Only:
    """Dataset with exactly Tier 1 fields and nothing else."""

    def test_tier1_only_is_valid(self):
        df = pd.DataFrame({
            "customer_id": ["A", "B"],
            "tenure_months": [12, 24],
            "monthly_charges": [50.0, 75.0],
            "total_charges": [600.0, 1800.0],
            "churn_label": [0, 1],
        })
        result = validate_dataset(df)

        assert result.is_valid is True

    def test_tier1_only_reports_all_tier2_missing(self):
        df = pd.DataFrame({
            "customer_id": ["A", "B"],
            "tenure_months": [12, 24],
            "monthly_charges": [50.0, 75.0],
            "total_charges": [600.0, 1800.0],
            "churn_label": [0, 1],
        })
        result = validate_dataset(df)

        assert set(result.tier2_missing) == set(TIER2_FIELDS)
        assert result.tier2_present == []

    def test_tier1_only_reports_all_tier3_missing(self):
        df = pd.DataFrame({
            "customer_id": ["A", "B"],
            "tenure_months": [12, 24],
            "monthly_charges": [50.0, 75.0],
            "total_charges": [600.0, 1800.0],
            "churn_label": [0, 1],
        })
        result = validate_dataset(df)

        assert set(result.tier3_missing) == set(TIER3_FIELDS)
        assert result.tier3_present == []


class TestCategoryValidation:
    """Fields with allowed_values should report violations."""

    def test_invalid_contract_type_produces_error(self):
        df = pd.DataFrame({
            "customer_id": ["A", "B"],
            "tenure_months": [12, 24],
            "monthly_charges": [50.0, 75.0],
            "total_charges": [600.0, 1800.0],
            "churn_label": [0, 1],
            "contract_type": ["month-to-month", "invalid_contract"],
        })
        result = validate_dataset(df)

        assert result.is_valid is True  # Tier 2 issues don't block
        assert any("contract_type" in err for err in result.errors)
        assert any("invalid_contract" in err for err in result.errors)

    def test_valid_contract_types_no_error(self):
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "tenure_months": [12, 24, 36],
            "monthly_charges": [50.0, 75.0, 100.0],
            "total_charges": [600.0, 1800.0, 3600.0],
            "churn_label": [0, 1, 0],
            "contract_type": ["month-to-month", "one_year", "two_year"],
        })
        result = validate_dataset(df)

        assert result.is_valid is True
        assert result.errors == []


class TestExtraColumns:
    """Extra columns not in the schema should be silently ignored."""

    def test_extra_columns_dont_affect_validity(self):
        df = pd.DataFrame({
            "customer_id": ["A", "B"],
            "tenure_months": [12, 24],
            "monthly_charges": [50.0, 75.0],
            "total_charges": [600.0, 1800.0],
            "churn_label": [0, 1],
            "random_extra_column": ["x", "y"],
            "another_one": [99, 100],
        })
        result = validate_dataset(df)

        assert result.is_valid is True
        assert result.errors == []
