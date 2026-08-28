"""
Mapping Config — The Rosetta Stone Between Client-Speak and Pipeline-Speak
==========================================================================

Every client calls their data something different. One says "MonthlyCharges",
another says "mrr", a third says "monthly_fee". They all mean the same thing:
how much the customer pays each month.

The mapping config is a translation dictionary — a Rosetta Stone that lets us
convert any client's column names, value formats, and data types into the
standard language our pipeline understands.

It has three layers of translation:
1. **Column mappings:** Rename columns (e.g., "MonthlyCharges" → "monthly_charges")
2. **Value mappings:** Convert values within a column (e.g., "Yes" → 1, "No" → 0)
3. **Type coercions:** Force a column to a specific type (e.g., "total_charges" → float)

Each client gets one YAML file. Once approved, it never changes unless the client's
data format changes. Think of it like programming a universal remote — you do the
setup once, then it just works every time.
"""

from dataclasses import dataclass, field
from typing import Any, Dict

import pandas as pd
import yaml


@dataclass
class MappingConfig:
    """
    A complete translation specification for one client's data.

    This is the full set of instructions for converting raw client data
    into the standardized format the pipeline expects. Like a recipe card:
    step 1 (rename columns), step 2 (convert values), step 3 (fix types).

    Attributes:
        client_id: Unique identifier for this client (e.g., "telco_ibm").
        source_description: Human-readable description of where this data comes from.
        column_mappings: Dict mapping raw column names to standard field names.
        value_mappings: Dict of field-level value conversions.
            Structure: {standard_field: {raw_value: standard_value}}
        type_coercions: Dict of fields that need explicit type casting.
            Structure: {standard_field: target_type_string}
    """

    client_id: str
    source_description: str = ""
    column_mappings: Dict[str, str] = field(default_factory=dict)
    value_mappings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    type_coercions: Dict[str, str] = field(default_factory=dict)


DRAFT_SUFFIX = ".draft.yaml"


class MappingNotApprovedError(ValueError):
    """
    Raised when a mapping config exists but no human has signed it off.

    This is the one failure in the LLM path that nothing downstream can catch. A
    wrong mapping does not raise: it renames a column to the wrong standard field,
    row counts still match, validation still passes, and the model trains on a
    feature that means something other than its name. So the gate has to be here,
    before the config is ever applied to data.
    """


def draft_reason(yaml_path: str, raw: Dict[str, Any] = None) -> str:
    """
    Explain why a config counts as an unapproved draft, or return "" if it is fine.

    Two independent signals, because either one alone is easy to defeat by
    accident: renaming the file without reading it, or editing it without
    renaming. A config with no 'status' key at all is treated as approved, which
    is what keeps every hand-written config from Chapter 1 working.

    Args:
        yaml_path: Path the config was loaded from. The filename is half the check.
        raw: Already-parsed YAML, if you have it. Read from disk otherwise.

    Returns:
        A human-readable reason, or "" when the config is approved.
    """
    if yaml_path.endswith(DRAFT_SUFFIX):
        return f"its filename ends in {DRAFT_SUFFIX}"

    if raw is None:
        try:
            with open(yaml_path, "r") as f:
                raw = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            # Unreadable is a different problem, and load_mapping_config reports it.
            return ""

    if str(raw.get("status", "")).lower() == "draft":
        return "it still says 'status: draft' inside"

    return ""


def load_mapping_config(yaml_path: str) -> MappingConfig:
    """
    Parse a YAML mapping file into a MappingConfig object.

    This reads the client's Rosetta Stone from disk. The YAML format is designed
    to be human-editable — a data engineer can write one by hand, or the LLM
    auto-mapping module can generate a draft for human review.

    Args:
        yaml_path: Path to the YAML mapping file.

    Returns:
        A fully populated MappingConfig ready to be applied to a DataFrame.

    Raises:
        FileNotFoundError: If the YAML file doesn't exist.
        yaml.YAMLError: If the file isn't valid YAML.
        KeyError: If required field 'client_id' is missing.
        MappingNotApprovedError: If the config is still a draft. There is no flag
            to skip this. An unreviewed mapping is the one thing in this pipeline
            that fails silently, so the only way past it is to review it.
    """
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Empty YAML file: {yaml_path}")

    if "client_id" not in raw:
        raise KeyError("Mapping config must contain 'client_id'")

    reason = draft_reason(yaml_path, raw)
    if reason:
        raise MappingNotApprovedError(
            f"Refusing to load {yaml_path}: {reason}. "
            "An LLM-drafted mapping has to be reviewed by a person before the "
            "pipeline uses it. Read the column mappings, fix what is wrong, set "
            "'status: approved', and rename the file to mapping.yaml."
        )

    return MappingConfig(
        client_id=raw["client_id"],
        source_description=raw.get("source_description", ""),
        column_mappings=raw.get("column_mappings", {}),
        value_mappings=raw.get("value_mappings", {}),
        type_coercions=raw.get("type_coercions", {}),
    )


def serialize_mapping_config(config: MappingConfig) -> str:
    """
    Serialize a MappingConfig object back to a YAML string.

    This is the reverse of load_mapping_config. Useful for:
    - Writing LLM-generated draft configs to disk
    - Programmatically creating configs for testing
    - Round-trip verification (serialize → parse → compare)

    Args:
        config: The MappingConfig to serialize.

    Returns:
        A YAML-formatted string representation of the config.
    """
    data: Dict[str, Any] = {
        "client_id": config.client_id,
        "source_description": config.source_description,
        "column_mappings": config.column_mappings,
        "value_mappings": config.value_mappings,
        "type_coercions": config.type_coercions,
    }

    # Remove empty optional sections to keep output clean
    if not data["source_description"]:
        del data["source_description"]
    if not data["column_mappings"]:
        del data["column_mappings"]
    if not data["value_mappings"]:
        del data["value_mappings"]
    if not data["type_coercions"]:
        del data["type_coercions"]

    return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)


def apply_mapping(df: pd.DataFrame, config: MappingConfig) -> pd.DataFrame:
    """
    Apply the full translation pipeline to a raw DataFrame.

    This is where the Rosetta Stone does its work. Three steps, in order:

    1. **Rename columns:** "MonthlyCharges" becomes "monthly_charges".
       Only renames columns that exist in the DataFrame — extras are ignored.

    2. **Map values:** Within specific columns, convert raw values to standard ones.
       Example: in the "churn_label" column, "Yes" becomes 1, "No" becomes 0.
       Only maps values that have explicit mappings — others pass through unchanged.

    3. **Coerce types:** Force columns to specific types.
       Example: "total_charges" might arrive as strings (because of " " values in
       the raw CSV). We coerce to float, turning unparseable values into NaN.

    Args:
        df: The raw DataFrame with client-native column names and values.
        config: The mapping config specifying all translations.

    Returns:
        A new DataFrame with standardized column names, values, and types.
        The original DataFrame is not modified.
    """
    # Work on a copy so we don't mutate the caller's data
    result = df.copy()

    # Step 1: Rename columns
    # Only rename columns that actually exist in this DataFrame
    rename_map = {
        raw_col: std_col
        for raw_col, std_col in config.column_mappings.items()
        if raw_col in result.columns
    }
    result = result.rename(columns=rename_map)

    # Step 2: Apply value mappings
    # For each field that has value mappings, replace matching values
    for field_name, value_map in config.value_mappings.items():
        if field_name in result.columns:
            result[field_name] = result[field_name].map(
                lambda x, vm=value_map: vm.get(x, x)
            )

    # Step 3: Coerce types
    # Convert columns to their target types, handling errors gracefully
    for field_name, target_type in config.type_coercions.items():
        if field_name not in result.columns:
            continue

        if target_type == "float":
            result[field_name] = pd.to_numeric(result[field_name], errors="coerce")
        elif target_type == "int":
            # First convert to numeric (handles strings), then to nullable int
            result[field_name] = pd.to_numeric(result[field_name], errors="coerce")
            # Only convert to int if there are no NaN values
            if not result[field_name].isna().any():
                result[field_name] = result[field_name].astype(int)
        elif target_type == "string":
            result[field_name] = result[field_name].astype(str)
        elif target_type == "category":
            result[field_name] = result[field_name].astype("category")

    return result
