"""
Dataset Validator — The Bouncer at the Door
============================================

Picture a nightclub with a strict dress code. The bouncer doesn't care if you're
wearing designer shoes (Tier 3) or a fancy watch (Tier 2) — those are nice but
optional. What they DO check is: do you have shoes at all? Do you have a shirt?
These are the non-negotiables. No shirt, no entry.

Our validator works the same way:

- **Tier 1 fields** are the non-negotiables. Without customer_id, we don't know
  WHO we're predicting for. Without churn_label, we have no ground truth to learn
  from. Without tenure/charges, we don't have the minimum signal to build a model.
  Missing ANY Tier 1 field = rejected at the door.

- **Tier 2/3 fields** are upgrades. Having contract_type makes the model smarter,
  having gender adds nuance — but the model can still function without them. Missing
  these is noted in the report (so we know what we're working with) but never
  causes rejection.

The validator also checks types: if tenure_months is supposed to be an integer but
contains "abc", that's an error. The data might have the right column name but
garbage inside it.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from churn_pipeline.data_contract import (
    STANDARD_SCHEMA,
    FieldSpec,
    Tier,
)


@dataclass
class ValidationResult:
    """
    The bouncer's verdict — did this dataset get in, and what's it wearing?

    Attributes:
        is_valid: True if ALL Tier 1 fields are present with acceptable types.
                  False if any Tier 1 field is missing.
        tier1_present: Names of Tier 1 fields found in the dataset.
        tier1_missing: Names of Tier 1 fields NOT found — these caused rejection.
        tier2_present: Names of Tier 2 fields found (bonus points).
        tier2_missing: Names of Tier 2 fields not found (noted, not penalized).
        tier3_present: Names of Tier 3 fields found (nice to have).
        tier3_missing: Names of Tier 3 fields not found (no penalty).
        errors: Specific problems found — type mismatches, empty values, etc.
    """

    is_valid: bool
    tier1_present: List[str] = field(default_factory=list)
    tier1_missing: List[str] = field(default_factory=list)
    tier2_present: List[str] = field(default_factory=list)
    tier2_missing: List[str] = field(default_factory=list)
    tier3_present: List[str] = field(default_factory=list)
    tier3_missing: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def validate_dataset(
    df: pd.DataFrame,
    schema: Dict[str, FieldSpec] = STANDARD_SCHEMA,
) -> ValidationResult:
    """
    Check incoming data against the data contract.

    The logic is simple:
    1. For each field in the schema, check if it exists in the DataFrame.
    2. Sort findings into tier1/tier2/tier3 present/missing buckets.
    3. If ANY Tier 1 field is missing → is_valid = False.
    4. For present fields, check that the dtype is compatible.

    This function does NOT transform the data — it only inspects and reports.
    Think of it as an X-ray: it tells you what's inside without changing anything.

    Args:
        df: The DataFrame to validate (should already have standardized column names
            after mapping config has been applied).
        schema: The data contract to validate against. Defaults to STANDARD_SCHEMA.

    Returns:
        A ValidationResult describing what was found, what's missing, and whether
        the dataset is acceptable for pipeline processing.
    """
    # Handle empty DataFrame — nothing to validate
    if df.empty:
        tier1_fields = [name for name, spec in schema.items() if spec.tier == Tier.REQUIRED]
        tier2_fields = [name for name, spec in schema.items() if spec.tier == Tier.ENGAGEMENT]
        tier3_fields = [name for name, spec in schema.items() if spec.tier == Tier.DEMOGRAPHIC]
        return ValidationResult(
            is_valid=False,
            tier1_present=[],
            tier1_missing=tier1_fields,
            tier2_present=[],
            tier2_missing=tier2_fields,
            tier3_present=[],
            tier3_missing=tier3_fields,
            errors=["DataFrame is empty (0 rows)"],
        )

    columns = set(df.columns)
    errors: List[str] = []

    tier1_present: List[str] = []
    tier1_missing: List[str] = []
    tier2_present: List[str] = []
    tier2_missing: List[str] = []
    tier3_present: List[str] = []
    tier3_missing: List[str] = []

    for field_name, spec in schema.items():
        if field_name in columns:
            # Field exists — check type compatibility
            _check_field_type(df, field_name, spec, errors)

            # Sort into the right bucket
            if spec.tier == Tier.REQUIRED:
                tier1_present.append(field_name)
            elif spec.tier == Tier.ENGAGEMENT:
                tier2_present.append(field_name)
            else:
                tier3_present.append(field_name)
        else:
            # Field missing — sort into the right bucket
            if spec.tier == Tier.REQUIRED:
                tier1_missing.append(field_name)
            elif spec.tier == Tier.ENGAGEMENT:
                tier2_missing.append(field_name)
            else:
                tier3_missing.append(field_name)

    # The gate decision: ALL Tier 1 fields must be present
    is_valid = len(tier1_missing) == 0

    return ValidationResult(
        is_valid=is_valid,
        tier1_present=tier1_present,
        tier1_missing=tier1_missing,
        tier2_present=tier2_present,
        tier2_missing=tier2_missing,
        tier3_present=tier3_present,
        tier3_missing=tier3_missing,
        errors=errors,
    )


def _check_field_type(
    df: pd.DataFrame,
    field_name: str,
    spec: FieldSpec,
    errors: List[str],
) -> None:
    """
    Verify that a column's actual dtype is compatible with what the schema expects.

    We're lenient here — we don't demand exact dtype matches. Instead we check
    that the data CAN be interpreted as the expected type. For example:
    - A column of int64 is fine for an "int" spec
    - A column of object (strings) is NOT fine for a "float" spec unless the
      strings are actually numeric

    Args:
        df: The DataFrame containing the column.
        field_name: Which column to check.
        spec: The FieldSpec defining what type we expect.
        errors: List to append error messages to (mutated in place).
    """
    col = df[field_name]
    actual_dtype = col.dtype

    if spec.dtype == "float":
        if not pd.api.types.is_numeric_dtype(actual_dtype):
            # Try converting — maybe strings that look like numbers
            try:
                converted = pd.to_numeric(col, errors="coerce")
                n_failed = converted.isna().sum() - col.isna().sum()
                if n_failed > 0:
                    errors.append(
                        f"Field '{field_name}': expected float, but {n_failed} values "
                        f"cannot be converted to numeric"
                    )
            except Exception:
                errors.append(
                    f"Field '{field_name}': expected float, got {actual_dtype}"
                )

    elif spec.dtype == "int":
        if not pd.api.types.is_numeric_dtype(actual_dtype):
            try:
                converted = pd.to_numeric(col, errors="coerce")
                n_failed = converted.isna().sum() - col.isna().sum()
                if n_failed > 0:
                    errors.append(
                        f"Field '{field_name}': expected int, but {n_failed} values "
                        f"cannot be converted to numeric"
                    )
            except Exception:
                errors.append(
                    f"Field '{field_name}': expected int, got {actual_dtype}"
                )

    elif spec.dtype == "category":
        # Check allowed values if specified
        if spec.allowed_values is not None:
            unique_vals = set(col.dropna().unique())
            allowed_set = set(spec.allowed_values)
            invalid_vals = unique_vals - allowed_set
            if invalid_vals:
                errors.append(
                    f"Field '{field_name}': found invalid values {invalid_vals}. "
                    f"Allowed: {spec.allowed_values}"
                )

    # "string" type — anything is acceptable, no check needed
