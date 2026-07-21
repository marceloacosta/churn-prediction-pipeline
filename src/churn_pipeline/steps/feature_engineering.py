"""
Feature Engineering — The Prep Chef
=====================================

Your model can't read English. It can't understand "contract = month-to-month"
or "payment_method = electronic_check". It only understands numbers — specifically,
grids of numbers where every cell is filled in.

Feature engineering is the prep chef that takes raw, validated ingredients and
chops them into a form the model can digest:

1. **Imputation (filling blanks):** Some values are missing. A model can't learn
   from a blank cell. We fill numeric blanks with the median (the middle value —
   robust to outliers) and categorical blanks with the mode (the most common value).

2. **Encoding (categories → numbers):** "month-to-month" means nothing to math.
   We assign each category a number: month-to-month=0, one_year=1, two_year=2.
   The model can now do arithmetic with contract types.

3. **Scaling (leveling the playing field):** If tenure ranges 1-72 and
   monthly_charges ranges 18-118, the model might think charges are "bigger"
   just because the numbers are larger. Scaling puts everything on the same
   0-to-1 scale so no feature dominates by accident.

4. **Interaction features (combinations that tell a story):** Monthly charges × tenure
   captures "total lifetime value" — something neither feature alone conveys. A customer
   paying $100/month for 36 months is very different from one paying $100/month for 2.

The key distinction: **training mode** (fit=True) learns the recipes from data,
while **scoring mode** (fit=False) applies previously-learned recipes without change.
This separation is critical — scoring data MUST be transformed the exact same way
as training data, or the model sees a different "language" than it learned.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from churn_pipeline.data_contract import STANDARD_SCHEMA, Tier


@dataclass
class FeatureArtifacts:
    """
    Stores all the 'recipes' learned during training so we can apply the
    exact same transformations during scoring.

    Think of it like a recipe card: during training, the chef figures out
    "median tenure is 29 months" and "there are 3 contract types." During
    scoring, the chef follows that same card without re-measuring anything.

    Attributes:
        scaler: Fitted MinMaxScaler — knows the min/max of each numeric feature.
        encoders: Dict of fitted LabelEncoders — one per categorical column,
            knows the mapping from category strings to integers.
        impute_values: Dict of values to fill missing data with.
            Numeric fields get the median, categorical get the mode.
        feature_names: Ordered list of final column names after transformation.
            This defines the exact column order the model expects.
    """

    scaler: Optional[MinMaxScaler] = None
    encoders: Dict[str, LabelEncoder] = field(default_factory=dict)
    impute_values: Dict[str, object] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)


# Fields we use for features (everything except the ID and the label)
_FEATURE_FIELDS = {
    name: spec
    for name, spec in STANDARD_SCHEMA.items()
    if name not in ("customer_id", "churn_label")
}

# Numeric fields (int or float in the schema)
_NUMERIC_FIELDS = [
    name for name, spec in _FEATURE_FIELDS.items() if spec.dtype in ("int", "float")
]

# Categorical fields
_CATEGORICAL_FIELDS = [
    name for name, spec in _FEATURE_FIELDS.items() if spec.dtype == "category"
]


def engineer_features(
    df: pd.DataFrame,
    fit: bool = True,
    artifacts: Optional[FeatureArtifacts] = None,
) -> Tuple[np.ndarray, FeatureArtifacts]:
    """
    Transform a validated DataFrame into a model-ready numeric matrix.

    This is the core transformation: raw columns go in, a clean grid of
    numbers comes out — no nulls, no strings, every column on the same scale.

    Two modes:
    - **fit=True (training):** Learn the transformation rules from this data.
      "What's the median tenure? What categories exist for contract_type?"
      Creates new FeatureArtifacts from scratch.

    - **fit=False (scoring):** Apply previously-learned rules without re-learning.
      Critical because the model expects data transformed the SAME way it was
      trained on. If training scaled tenure to [0,1] using min=1, max=72, then
      scoring must use those same min/max values — even if new data ranges 1-84.

    Args:
        df: Validated DataFrame with standardized column names.
            Must contain at least the Tier 1 numeric fields.
        fit: If True, learn new transformation parameters from this data.
            If False, use the provided artifacts (required).
        artifacts: Pre-fitted FeatureArtifacts to apply in scoring mode.
            Required when fit=False. Ignored when fit=True.

    Returns:
        Tuple of (feature_matrix, artifacts) where:
        - feature_matrix: np.ndarray of shape (n_rows, n_features), dtype float64,
          guaranteed to have zero NaN values.
        - artifacts: The FeatureArtifacts used (new if fit=True, same object if fit=False).

    Raises:
        ValueError: If fit=False and no artifacts are provided.
    """
    if not fit and artifacts is None:
        raise ValueError(
            "Scoring mode (fit=False) requires pre-fitted artifacts. "
            "Run in training mode first to create them."
        )

    if fit:
        artifacts = FeatureArtifacts()

    # Work on a copy — never mutate the caller's DataFrame
    work = df.copy()

    # Identify which feature columns actually exist in this DataFrame
    numeric_cols = [c for c in _NUMERIC_FIELDS if c in work.columns]
    categorical_cols = [c for c in _CATEGORICAL_FIELDS if c in work.columns]

    # --- Step 1: Imputation (fill the blanks) ---
    work = _impute(work, numeric_cols, categorical_cols, fit, artifacts)

    # --- Step 2: Encode categoricals (strings → numbers) ---
    work = _encode_categoricals(work, categorical_cols, fit, artifacts)

    # --- Step 3: Create interaction features ---
    work = _create_interactions(work, numeric_cols)

    # --- Step 4: Assemble final feature matrix ---
    # Determine the ordered list of feature columns
    interaction_cols = [c for c in work.columns if c.startswith("interaction_")]
    all_feature_cols = numeric_cols + categorical_cols + interaction_cols

    if fit:
        artifacts.feature_names = all_feature_cols

    # Use the artifact's feature order (handles case where scoring data
    # might have columns in different order)
    feature_cols_ordered = artifacts.feature_names

    # Ensure all expected columns exist (fill with 0 if a column is missing)
    for col in feature_cols_ordered:
        if col not in work.columns:
            work[col] = 0.0

    feature_matrix = work[feature_cols_ordered].values.astype(np.float64)

    # --- Step 5: Scale to [0, 1] ---
    feature_matrix = _scale(feature_matrix, fit, artifacts)

    return feature_matrix, artifacts


def _impute(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    fit: bool,
    artifacts: FeatureArtifacts,
) -> pd.DataFrame:
    """
    Fill missing values — numeric columns get the median, categoricals get the mode.

    Why median instead of mean? Because median is robust to outliers. If one customer
    has $10,000 monthly charges (data error), the mean gets pulled way up, but the
    median barely moves. For categorical, mode (most common value) is the safest guess.
    """
    if fit:
        # Learn imputation values from the data
        for col in numeric_cols:
            if col in df.columns:
                median_val = df[col].median()
                # If all values are NaN, use 0 as fallback
                artifacts.impute_values[col] = 0.0 if pd.isna(median_val) else float(median_val)

        for col in categorical_cols:
            if col in df.columns:
                mode_series = df[col].mode()
                artifacts.impute_values[col] = mode_series.iloc[0] if len(mode_series) > 0 else "unknown"

    # Apply imputation
    for col in numeric_cols:
        if col in df.columns and col in artifacts.impute_values:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(artifacts.impute_values[col])

    for col in categorical_cols:
        if col in df.columns and col in artifacts.impute_values:
            df[col] = df[col].fillna(artifacts.impute_values[col])

    return df


def _encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: List[str],
    fit: bool,
    artifacts: FeatureArtifacts,
) -> pd.DataFrame:
    """
    Convert category strings to integers.

    Each category gets a unique number: month-to-month=0, one_year=1, two_year=2.
    The encoder remembers this mapping so scoring data gets the same numbers.

    Unknown categories (not seen during training) are assigned -1, then imputed to 0.
    """
    for col in categorical_cols:
        if col not in df.columns:
            continue

        if fit:
            encoder = LabelEncoder()
            # Fit on string values, handling any remaining NaN
            values = df[col].astype(str).values
            encoder.fit(values)
            artifacts.encoders[col] = encoder
            df[col] = encoder.transform(values)
        else:
            if col in artifacts.encoders:
                encoder = artifacts.encoders[col]
                values = df[col].astype(str).values
                # Handle unseen categories gracefully
                known_classes = set(encoder.classes_)
                safe_values = np.array(
                    [v if v in known_classes else encoder.classes_[0] for v in values]
                )
                df[col] = encoder.transform(safe_values)
            else:
                # Column exists in data but wasn't in training — encode as 0
                df[col] = 0

    return df


def _create_interactions(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """
    Create interaction features — combinations that capture relationships
    individual features miss.

    monthly_charges × tenure_months = approximate total lifetime value.
    A customer paying $100/month for 36 months has a very different relationship
    with the company than one paying $100/month for 2 months, even though their
    monthly_charges are identical.
    """
    if "monthly_charges" in df.columns and "tenure_months" in df.columns:
        df["interaction_charges_x_tenure"] = (
            df["monthly_charges"].astype(float) * df["tenure_months"].astype(float)
        )

    return df


def _scale(
    matrix: np.ndarray,
    fit: bool,
    artifacts: FeatureArtifacts,
) -> np.ndarray:
    """
    Scale all features to [0, 1] range using Min-Max scaling.

    Why scale? If tenure ranges 1-72 and monthly_charges ranges 18-118,
    a model might over-weight charges simply because the numbers are bigger.
    Scaling puts everything on equal footing — a value of 0.5 means "halfway
    between the lowest and highest seen during training" regardless of the
    original unit.
    """
    if fit:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(matrix)
        artifacts.scaler = scaler
    else:
        scaled = artifacts.scaler.transform(matrix)

    # Clip to [0, 1] — scoring data might exceed training range
    scaled = np.clip(scaled, 0.0, 1.0)

    # Final safety: replace any lingering NaN with 0
    scaled = np.nan_to_num(scaled, nan=0.0)

    return scaled
