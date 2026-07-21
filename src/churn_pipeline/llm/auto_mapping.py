"""
LLM Auto-Mapping — Teaching a Machine to Read Column Names
============================================================

When a new client uploads a CSV, someone has to figure out what each column means.
A human looks at "MonthlyCharges" and immediately knows it maps to "monthly_charges."
They see "CustID" and know it's "customer_id." This is fundamentally a natural
language problem — recognizing that different words mean the same thing.

LLMs are trained on exactly this kind of pattern recognition. We send Claude the
column names + a few sample rows, and it drafts a mapping YAML. A human reviews
and approves it before the pipeline trusts it.

**Why an LLM instead of rules?** A rule-based approach would need an ever-growing
dictionary of synonyms. "MonthlyCharges", "mrr", "monthly_fee", "amt_per_month",
"recurring_revenue" — the list never ends. An LLM handles all of these because it
understands language, not just exact matches.

**This step is non-blocking:** if Bedrock fails, the pipeline continues without
auto-mapping. The operator writes the YAML manually. No data is lost.
"""

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import yaml

from churn_pipeline.data_contract import STANDARD_SCHEMA

logger = logging.getLogger(__name__)


@dataclass
class ColumnMapping:
    """
    A single mapping suggestion from the LLM.

    Attributes:
        source_column: Original column name in the client's CSV.
        target_field: Standard field name from our data contract.
        confidence: How sure the LLM is — "high", "medium", or "low".
        reasoning: Why the LLM thinks this mapping is correct.
    """

    source_column: str
    target_field: str
    confidence: str  # "high", "medium", "low"
    reasoning: str


@dataclass
class AutoMappingResult:
    """
    The complete output of the auto-mapping process.

    Attributes:
        client_id: Which client this mapping is for.
        source_columns: All columns found in the CSV.
        sample_rows: A few example rows for context.
        mappings: The LLM's mapping suggestions.
        unmapped_columns: Columns the LLM couldn't map to any standard field.
        draft_yaml_path: Where the .draft.yaml was written.
    """

    client_id: str
    source_columns: List[str]
    sample_rows: List[Dict]
    mappings: List[ColumnMapping]
    unmapped_columns: List[str]
    draft_yaml_path: str


def read_csv_metadata(
    s3_path: str,
    sample_size: int = 5,
    boto3_client=None,
) -> Tuple[List[str], List[Dict]]:
    """
    Read column names and up to sample_size rows from a CSV in S3.

    This gives the LLM enough context to make mapping guesses — it sees
    the column names AND actual data values (which often disambiguate:
    a column named "id" could be anything, but if values look like
    "CUST-7590-VHVEG", it's clearly a customer ID).

    Args:
        s3_path: Full s3:// path to the CSV file.
        sample_size: How many rows to sample (default: 5).
        boto3_client: Optional pre-configured S3 client (for testing).

    Returns:
        Tuple of (column_names, sample_rows) where sample_rows is a list
        of dicts (one per row, keys are column names).
    """
    import io

    import boto3
    import pandas as pd

    if boto3_client is None:
        boto3_client = boto3.client("s3")

    # Parse bucket and key from s3:// path
    path_parts = s3_path.replace("s3://", "").split("/", 1)
    bucket = path_parts[0]
    key = path_parts[1]

    # Read just the first few rows
    response = boto3_client.get_object(Bucket=bucket, Key=key)
    content = response["Body"].read().decode("utf-8")
    df = pd.read_csv(io.StringIO(content), nrows=sample_size)

    columns = df.columns.tolist()
    sample_rows = df.to_dict(orient="records")

    return columns, sample_rows


def build_mapping_prompt(
    columns: List[str],
    sample_rows: List[Dict],
    standard_fields: Dict = None,
) -> str:
    """
    Construct the prompt for Claude that asks it to map client columns
    to standard fields.

    The prompt includes:
    - All standard fields with their descriptions (what we're mapping TO)
    - The client's column names (what we're mapping FROM)
    - Sample data rows (for disambiguation)
    - Instructions to output structured JSON with confidence scores

    Args:
        columns: Client's CSV column names.
        sample_rows: A few example data rows (list of dicts).
        standard_fields: The data contract fields. Defaults to STANDARD_SCHEMA.

    Returns:
        The complete prompt string ready to send to Bedrock.
    """
    if standard_fields is None:
        standard_fields = STANDARD_SCHEMA

    # Build the standard fields section
    fields_description = []
    for name, spec in standard_fields.items():
        fields_description.append(
            f"  - {name} ({spec.dtype}): {spec.description}"
        )
    fields_text = "\n".join(fields_description)

    # Build sample data section
    sample_text = ""
    if sample_rows:
        for i, row in enumerate(sample_rows[:3]):
            row_str = ", ".join(f"{k}={v!r}" for k, v in list(row.items())[:8])
            sample_text += f"  Row {i + 1}: {row_str}\n"

    prompt = f"""You are a data mapping expert. A client has uploaded a CSV with the following columns:

Client columns: {columns}

Sample data:
{sample_text}

Your task: Map each client column to the most appropriate standard field from this list:

Standard fields:
{fields_text}

For each mapping, provide:
1. The source column name (from the client)
2. The target field name (from the standard list)
3. A confidence level: "high", "medium", or "low"
4. A brief reasoning why you think this mapping is correct

Output your response as a JSON array of objects with keys: "source_column", "target_field", "confidence", "reasoning"

If a client column doesn't map to any standard field, include it with target_field: null.

Only map columns where you have reasonable confidence. It's better to leave a column unmapped than to guess wrong."""

    return prompt


def call_bedrock_for_mapping(
    prompt: str,
    boto3_client=None,
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
) -> Optional[List[ColumnMapping]]:
    """
    Call Amazon Bedrock (Claude) with the mapping prompt.

    Returns parsed mappings or None if the call fails.
    Non-blocking: failures are logged and the function returns None.

    Args:
        prompt: The mapping prompt (from build_mapping_prompt).
        boto3_client: Optional pre-configured Bedrock client (for testing).
        model_id: Which Claude model to use.

    Returns:
        List of ColumnMapping objects, or None if the call failed.
    """
    import boto3

    try:
        if boto3_client is None:
            boto3_client = boto3.client("bedrock-runtime")

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        })

        response = boto3_client.invoke_model(
            modelId=model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )

        response_body = json.loads(response["body"].read())
        response_text = response_body["content"][0]["text"]

        # Parse the JSON response
        return _parse_mapping_response(response_text)

    except Exception as e:
        logger.warning(f"Bedrock auto-mapping call failed: {e}")
        return None


def _parse_mapping_response(response_text: str) -> Optional[List[ColumnMapping]]:
    """Parse the LLM response text into ColumnMapping objects."""
    try:
        # Try to extract JSON from the response
        # The LLM might wrap it in markdown code blocks
        text = response_text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        mappings_data = json.loads(text)

        mappings = []
        for item in mappings_data:
            if item.get("target_field") is not None:
                mappings.append(ColumnMapping(
                    source_column=item["source_column"],
                    target_field=item["target_field"],
                    confidence=item.get("confidence", "medium"),
                    reasoning=item.get("reasoning", ""),
                ))

        return mappings

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Failed to parse LLM mapping response: {e}")
        return None


def write_draft_yaml(
    mappings: List[ColumnMapping],
    client_id: str,
    output_path: str,
) -> str:
    """
    Write the LLM's mapping suggestions as a .draft.yaml file.

    Includes confidence scores as comments for human review. The human
    reads this, fixes any mistakes, and renames it from .draft.yaml to
    mapping.yaml to approve it.

    Args:
        mappings: The LLM's mapping suggestions.
        client_id: Client identifier.
        output_path: Where to write the draft file.

    Returns:
        The path where the draft was written.
    """
    # Build the mapping config structure
    column_mappings = {}
    confidence_scores = {}

    for m in mappings:
        column_mappings[m.source_column] = m.target_field
        confidence_scores[m.source_column] = m.confidence

    config_data = {
        "client_id": client_id,
        "source_description": "Auto-generated by LLM from CSV analysis",
        "status": "draft",
        "column_mappings": column_mappings,
        "confidence_scores": confidence_scores,
        "value_mappings": {},
        "type_coercions": {},
    }

    yaml_content = yaml.dump(config_data, default_flow_style=False, sort_keys=False)

    # Add confidence as inline comments for readability
    lines = yaml_content.split("\n")
    output_lines = []
    for line in lines:
        output_lines.append(line)

    with open(output_path, "w") as f:
        f.write("\n".join(output_lines))

    return output_path


def is_mapping_approved(config_path: str) -> bool:
    """
    Check if a mapping.yaml (not .draft.yaml) exists at the config path.

    The pipeline only proceeds with a mapping config if the human has
    approved it. Approval = the file exists as 'mapping.yaml' (not
    'mapping.draft.yaml'). The human reviews the draft and renames it.

    Args:
        config_path: Path to check (should end in mapping.yaml).

    Returns:
        True if the approved mapping file exists. False if only a draft
        exists or nothing exists.
    """
    import os

    # The approved config must be exactly 'mapping.yaml', not '.draft.yaml'
    if config_path.endswith(".draft.yaml"):
        return False

    return os.path.exists(config_path)
