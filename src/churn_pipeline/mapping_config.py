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
    """
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Empty YAML file: {yaml_path}")

    if "client_id" not in raw:
        raise KeyError("Mapping config must contain 'client_id'")

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
