"""
Data Cleaning — The Janitor That Mops Up Before the Bouncer Checks IDs
========================================================================

Real-world data is messy. Not "a few blank cells" messy — we're talking:

- A column that's SUPPOSED to be numbers but has "N/A", "---", "ERROR", and "null"
  sprinkled randomly among real values
- The same category spelled 5 different ways: "Month-to-month", "month-to-month",
  "m2m", "MTM", "Month to Month"
- Duplicate rows (same customer appearing twice because someone merged CSVs badly)
- Outliers that are clearly data errors (a $99,999 monthly charge, or -5 tenure months)
- Mixed problems in the SAME column: some rows are valid, some are text garbage,
  some are blank, some are negative numbers that make no sense

The validator (Chapter 2) checks IF the right columns exist. But it doesn't FIX
problems — it just reports them. This cleaning step actually repairs what it can
and reports what it couldn't.

**The order is:** Map → Clean → Validate → Feature Engineer

Think of it like a restaurant kitchen:
- Mapping = the delivery truck brings ingredients labeled in a foreign language,
  we relabel them in English
- Cleaning = the prep cook washes vegetables, trims rotten bits, throws out
  anything that's clearly not food
- Validation = the head chef inspects: "Do we have all the ingredients we need?"
- Feature engineering = the sous chef chops everything into the right sizes
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd


@dataclass
class CleaningReport:
    """
    What the janitor found and did.

    This report tells you exactly what was wrong with the data and what
    was done about it. Unlike a silent data pipeline that hides problems,
    this makes every decision visible.

    Attributes:
        rows_before: How many rows we started with.
        rows_after: How many rows after cleaning (may be less if duplicates removed).
        duplicates_removed: Number of exact duplicate rows dropped.
        outliers_capped: Dict of {column: count} — how many values were capped per column.
        garbage_values: Dict of {column: {count, percentage, examples}} — non-parseable values found.
        categories_normalized: Dict of {column: {original: normalized}} — fuzzy matches applied.
        nulls_found: Dict of {column: count} — NaN values per column after cleaning.
        warnings: List of human-readable warnings about data quality issues.
    """

    rows_before: int = 0
    rows_after: int = 0
    duplicates_removed: int = 0
    outliers_capped: Dict[str, int] = field(default_factory=dict)
    garbage_values: Dict[str, Dict] = field(default_factory=dict)
    categories_normalized: Dict[str, Dict[str, str]] = field(default_factory=dict)
    nulls_found: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Known category synonyms — different ways people write the same thing
# ---------------------------------------------------------------------------
# The left side is what we might see in messy data.
# The right side is what our pipeline expects.
# This gets expanded over time as we encounter new clients.

CATEGORY_SYNONYMS = {
    "contract_type": {
        # Standard values: "month-to-month", "one_year", "two_year"
        "month to month": "month-to-month",
        "month-to-month": "month-to-month",
        "m2m": "month-to-month",
        "mtm": "month-to-month",
        "monthly": "month-to-month",
        "one year": "one_year",
        "one_year": "one_year",
        "1 year": "one_year",
        "1year": "one_year",
        "12 months": "one_year",
        "annual": "one_year",
        "two year": "two_year",
        "two_year": "two_year",
        "2 year": "two_year",
        "2year": "two_year",
        "24 months": "two_year",
        "biennial": "two_year",
    },
    "payment_method": {
        "credit card": "credit_card",
        "credit_card": "credit_card",
        "cc": "credit_card",
        "visa": "credit_card",
        "mastercard": "credit_card",
        "bank transfer": "bank_transfer",
        "bank_transfer": "bank_transfer",
        "ach": "bank_transfer",
        "direct debit": "bank_transfer",
        "electronic check": "electronic_check",
        "electronic_check": "electronic_check",
        "e-check": "electronic_check",
        "echeck": "electronic_check",
        "mailed check": "mailed_check",
        "mailed_check": "mailed_check",
        "check": "mailed_check",
    },
    "gender": {
        "male": "Male",
        "m": "Male",
        "female": "Female",
        "f": "Female",
    },
}

# Reasonable ranges for numeric columns — values outside these are likely errors
NUMERIC_BOUNDS = {
    "tenure_months": {"min": 0, "max": 120},       # 0 to 10 years
    "monthly_charges": {"min": 0, "max": 500},     # $0 to $500/month
    "total_charges": {"min": 0, "max": 50000},     # $0 to $50K lifetime
    "support_tickets": {"min": 0, "max": 100},     # 0 to 100 tickets
}


def clean_dataset(
    df: pd.DataFrame,
    remove_duplicates: bool = True,
    cap_outliers: bool = True,
    normalize_categories: bool = True,
    max_garbage_pct: float = 0.50,
) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Clean a mapped DataFrame before validation.

    This does NOT silently fix things — it documents every change in the
    CleaningReport so you know exactly what was wrong and what was done.

    The philosophy: fix what's clearly fixable (typos, duplicates, outliers),
    turn unparseable garbage into NaN (so imputation handles it later),
    and WARN about anything concerning.

    Args:
        df: DataFrame after mapping config has been applied (standard column names).
        remove_duplicates: Whether to drop exact duplicate rows.
        cap_outliers: Whether to cap values outside reasonable bounds.
        normalize_categories: Whether to apply fuzzy category matching.
        max_garbage_pct: If more than this % of a column is garbage, warn loudly.

    Returns:
        Tuple of (cleaned_dataframe, cleaning_report).
    """
    report = CleaningReport(rows_before=len(df))
    result = df.copy()

    # Step 1: Remove exact duplicate rows
    if remove_duplicates:
        result, report = _remove_duplicates(result, report)

    # Step 2: Clean numeric columns (coerce garbage to NaN, cap outliers)
    result, report = _clean_numeric_columns(result, report, cap_outliers, max_garbage_pct)

    # Step 3: Normalize categorical values (fuzzy matching)
    if normalize_categories:
        result, report = _normalize_categories(result, report)

    # Step 4: Report remaining nulls
    for col in result.columns:
        null_count = int(result[col].isna().sum())
        if null_count > 0:
            report.nulls_found[col] = null_count

    report.rows_after = len(result)

    # Summary warnings
    total_nulls = sum(report.nulls_found.values())
    if total_nulls > 0:
        null_pct = total_nulls / (len(result) * len(result.columns)) * 100
        report.warnings.append(
            f"Total null values after cleaning: {total_nulls} "
            f"({null_pct:.1f}% of all cells). "
            f"These will be filled by imputation in feature engineering."
        )

    return result, report


def _remove_duplicates(df: pd.DataFrame, report: CleaningReport) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Drop duplicate rows. Checks two types:
    1. Exact duplicates (every column identical)
    2. Duplicate customer_ids (same customer appearing multiple times)

    Why this happens: someone merges two CSV exports that overlap, or a system
    logs the same event twice. Duplicates inflate the dataset and can bias
    the model toward whatever was duplicated.
    """
    n_before = len(df)

    # First: exact row duplicates
    df = df.drop_duplicates()
    exact_dupes = n_before - len(df)

    # Second: duplicate customer_ids (keep first occurrence)
    id_dupes = 0
    if "customer_id" in df.columns:
        n_before_id = len(df)
        df = df.drop_duplicates(subset=["customer_id"], keep="first")
        id_dupes = n_before_id - len(df)

    n_removed = exact_dupes + id_dupes

    if n_removed > 0:
        report.duplicates_removed = n_removed
        parts = []
        if exact_dupes > 0:
            parts.append(f"{exact_dupes} exact duplicate rows")
        if id_dupes > 0:
            parts.append(f"{id_dupes} duplicate customer_ids (kept first occurrence)")
        report.warnings.append(
            f"Removed {n_removed} duplicates: {', '.join(parts)} "
            f"({n_removed / n_before * 100:.1f}% of dataset)."
        )

    return df, report


def _clean_numeric_columns(
    df: pd.DataFrame,
    report: CleaningReport,
    cap_outliers: bool,
    max_garbage_pct: float,
) -> tuple[pd.DataFrame, CleaningReport]:
    """
    For each numeric column:
    1. Try to convert everything to numbers (garbage → NaN)
    2. Report how much garbage was found
    3. Cap outliers to reasonable bounds

    This handles the real-world mess: columns that are SUPPOSED to be numbers
    but contain "N/A", "ERROR", "---", empty strings, "null", etc.
    """
    numeric_cols = [
        col for col in NUMERIC_BOUNDS.keys()
        if col in df.columns
    ]

    for col in numeric_cols:
        # Step A: Identify what's currently NOT a number
        original = df[col].copy()

        # Try to convert to numeric — anything that can't be parsed becomes NaN
        numeric_version = pd.to_numeric(df[col], errors="coerce")

        # Count garbage: values that were NOT NaN before but ARE NaN after conversion
        was_not_null = original.notna()
        is_now_null = numeric_version.isna()
        garbage_mask = was_not_null & is_now_null
        garbage_count = int(garbage_mask.sum())

        if garbage_count > 0:
            garbage_pct = garbage_count / len(df) * 100
            # Collect examples of the garbage values
            garbage_examples = original[garbage_mask].unique()[:5].tolist()

            report.garbage_values[col] = {
                "count": garbage_count,
                "percentage": round(garbage_pct, 1),
                "examples": [str(x) for x in garbage_examples],
            }

            if garbage_pct > max_garbage_pct * 100:
                report.warnings.append(
                    f"CRITICAL: Column '{col}' has {garbage_pct:.1f}% garbage values "
                    f"(examples: {garbage_examples[:3]}). "
                    f"This column may be unreliable."
                )
            else:
                report.warnings.append(
                    f"Column '{col}': {garbage_count} values ({garbage_pct:.1f}%) "
                    f"couldn't be parsed as numbers (examples: {garbage_examples[:3]}). "
                    f"Converted to NaN — will be imputed later."
                )

        # Apply the numeric conversion
        df[col] = numeric_version

        # Step B: Cap outliers to reasonable bounds
        if cap_outliers and col in NUMERIC_BOUNDS:
            bounds = NUMERIC_BOUNDS[col]
            lower, upper = bounds["min"], bounds["max"]

            below = (df[col] < lower) & df[col].notna()
            above = (df[col] > upper) & df[col].notna()
            n_capped = int(below.sum() + above.sum())

            if n_capped > 0:
                df.loc[below, col] = lower
                df.loc[above, col] = upper
                report.outliers_capped[col] = n_capped
                report.warnings.append(
                    f"Column '{col}': {n_capped} outlier values capped to "
                    f"[{lower}, {upper}] range."
                )

    return df, report


def _normalize_categories(
    df: pd.DataFrame,
    report: CleaningReport,
) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Normalize categorical values using fuzzy synonym matching.

    This handles the real-world problem where the same thing is spelled
    10 different ways: "Month-to-month", "month to month", "m2m", "MTM", etc.

    We lowercase everything and check against a known synonym dictionary.
    Values that don't match any known synonym are left as-is (they'll either
    be caught by validation or handled as "unknown" during encoding).
    """
    for col, synonyms in CATEGORY_SYNONYMS.items():
        if col not in df.columns:
            continue

        normalized_map = {}  # Track what we changed
        original = df[col].copy()

        def normalize_value(val):
            if pd.isna(val):
                return val
            val_lower = str(val).strip().lower()
            if val_lower in synonyms:
                return synonyms[val_lower]
            return val  # Unknown value — leave as-is

        df[col] = df[col].apply(normalize_value)

        # Report what changed
        changed_mask = (original != df[col]) & original.notna()
        if changed_mask.any():
            changes = {}
            for orig, new in zip(original[changed_mask], df[col][changed_mask]):
                if str(orig) not in changes:
                    changes[str(orig)] = str(new)

            report.categories_normalized[col] = changes

    return df, report


def print_cleaning_report(report: CleaningReport) -> None:
    """Pretty-print a cleaning report for notebook display."""
    print("DATA CLEANING REPORT")
    print("=" * 60)
    print(f"\n  Rows: {report.rows_before} → {report.rows_after}", end="")
    if report.duplicates_removed:
        print(f" ({report.duplicates_removed} duplicates removed)")
    else:
        print(" (no duplicates found)")

    if report.garbage_values:
        print(f"\n  Garbage values found (converted to NaN):")
        for col, info in report.garbage_values.items():
            print(f"    {col}: {info['count']} values ({info['percentage']}%)")
            print(f"      Examples: {info['examples'][:3]}")

    if report.outliers_capped:
        print(f"\n  Outliers capped:")
        for col, count in report.outliers_capped.items():
            bounds = NUMERIC_BOUNDS[col]
            print(f"    {col}: {count} values capped to [{bounds['min']}, {bounds['max']}]")

    if report.categories_normalized:
        print(f"\n  Categories normalized:")
        for col, changes in report.categories_normalized.items():
            print(f"    {col}:")
            for orig, new in list(changes.items())[:5]:
                print(f"      '{orig}' → '{new}'")

    if report.nulls_found:
        print(f"\n  Remaining nulls (will be imputed in feature engineering):")
        for col, count in report.nulls_found.items():
            pct = count / report.rows_after * 100
            print(f"    {col}: {count} ({pct:.1f}%)")

    if report.warnings:
        print(f"\n  Warnings:")
        for w in report.warnings:
            print(f"    ⚠ {w}")

    print()
