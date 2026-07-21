"""
Unit Tests: LLM Auto-Mapping Module
=====================================

Tests prompt construction, draft YAML output, approval check, and failure handling.

Validates: Requirements 1.1, 1.3, 1.4, 1.5, 1.6

# Feature: churn-prediction-pipeline, Task 12.2: Auto-mapping unit tests
"""

import os
import tempfile

import pytest

from churn_pipeline.data_contract import STANDARD_SCHEMA
from churn_pipeline.llm.auto_mapping import (
    ColumnMapping,
    _parse_mapping_response,
    build_mapping_prompt,
    is_mapping_approved,
    write_draft_yaml,
)


class TestPromptConstruction:
    """Verify the prompt includes all necessary information."""

    def test_prompt_includes_all_standard_field_names(self):
        """Every standard field must appear in the prompt."""
        columns = ["customerID", "tenure", "MonthlyCharges"]
        sample_rows = [{"customerID": "A", "tenure": 12, "MonthlyCharges": 50.0}]

        prompt = build_mapping_prompt(columns, sample_rows)

        for field_name in STANDARD_SCHEMA.keys():
            assert field_name in prompt, (
                f"Standard field '{field_name}' not found in prompt"
            )

    def test_prompt_includes_client_columns(self):
        """All client column names must appear in the prompt."""
        columns = ["CustID", "MonthlyFee", "ChurnStatus", "ContractLen"]
        sample_rows = [{"CustID": "X1", "MonthlyFee": 99.0}]

        prompt = build_mapping_prompt(columns, sample_rows)

        for col in columns:
            assert col in prompt, f"Client column '{col}' not found in prompt"

    def test_prompt_includes_sample_data(self):
        """Sample data values should be embedded in the prompt."""
        columns = ["id", "amount"]
        sample_rows = [
            {"id": "CUST-7590", "amount": 29.85},
            {"id": "CUST-5575", "amount": 56.95},
        ]

        prompt = build_mapping_prompt(columns, sample_rows)

        assert "CUST-7590" in prompt
        assert "29.85" in prompt

    def test_prompt_includes_field_descriptions(self):
        """Field descriptions from the data contract must appear for context."""
        columns = ["x"]
        sample_rows = []

        prompt = build_mapping_prompt(columns, sample_rows)

        # Check a few key descriptions are included
        assert "Unique customer identifier" in prompt
        assert "What the customer pays each month" in prompt


class TestDraftYAML:
    """Verify draft YAML output format."""

    def test_draft_yaml_includes_confidence_scores(self):
        """Draft YAML must include confidence scores."""
        mappings = [
            ColumnMapping("MonthlyCharges", "monthly_charges", "high", "Name matches"),
            ColumnMapping("CustID", "customer_id", "medium", "Likely an ID field"),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".draft.yaml", delete=False) as f:
            tmp_path = f.name

        try:
            write_draft_yaml(mappings, "test_client", tmp_path)

            with open(tmp_path) as f:
                content = f.read()

            assert "high" in content
            assert "medium" in content
            assert "confidence_scores" in content
            assert "test_client" in content
        finally:
            os.unlink(tmp_path)

    def test_draft_yaml_includes_column_mappings(self):
        """Draft YAML must contain the actual column mappings."""
        mappings = [
            ColumnMapping("tenure", "tenure_months", "high", "Direct match"),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".draft.yaml", delete=False) as f:
            tmp_path = f.name

        try:
            write_draft_yaml(mappings, "client_x", tmp_path)

            with open(tmp_path) as f:
                content = f.read()

            assert "tenure: tenure_months" in content
            assert "client_x" in content
        finally:
            os.unlink(tmp_path)


class TestMappingApproval:
    """Verify the approval check logic."""

    def test_draft_yaml_not_approved(self):
        """A .draft.yaml file should NOT be considered approved."""
        assert is_mapping_approved("/some/path/mapping.draft.yaml") is False

    def test_mapping_yaml_approved_when_exists(self):
        """mapping.yaml is approved if the file exists."""
        with tempfile.NamedTemporaryFile(suffix="mapping.yaml", delete=False) as f:
            tmp_path = f.name

        try:
            assert is_mapping_approved(tmp_path) is True
        finally:
            os.unlink(tmp_path)

    def test_mapping_yaml_not_approved_when_missing(self):
        """mapping.yaml is not approved if the file doesn't exist."""
        assert is_mapping_approved("/nonexistent/path/mapping.yaml") is False


class TestResponseParsing:
    """Verify LLM response parsing handles various formats."""

    def test_parse_valid_json_response(self):
        """Valid JSON array should parse into ColumnMapping list."""
        response = """[
            {"source_column": "MonthlyCharges", "target_field": "monthly_charges", "confidence": "high", "reasoning": "Name match"},
            {"source_column": "CustID", "target_field": "customer_id", "confidence": "medium", "reasoning": "ID field"}
        ]"""

        result = _parse_mapping_response(response)
        assert result is not None
        assert len(result) == 2
        assert result[0].source_column == "MonthlyCharges"
        assert result[0].target_field == "monthly_charges"
        assert result[0].confidence == "high"

    def test_parse_json_in_code_block(self):
        """JSON wrapped in markdown code blocks should still parse."""
        response = """Here's the mapping:

```json
[
    {"source_column": "tenure", "target_field": "tenure_months", "confidence": "high", "reasoning": "Direct"}
]
```
"""

        result = _parse_mapping_response(response)
        assert result is not None
        assert len(result) == 1
        assert result[0].target_field == "tenure_months"

    def test_parse_null_target_excluded(self):
        """Columns with null target_field should be excluded from mappings."""
        response = """[
            {"source_column": "MonthlyCharges", "target_field": "monthly_charges", "confidence": "high", "reasoning": "Match"},
            {"source_column": "RandomCol", "target_field": null, "confidence": "low", "reasoning": "No match"}
        ]"""

        result = _parse_mapping_response(response)
        assert result is not None
        assert len(result) == 1  # null target excluded

    def test_parse_malformed_response_returns_none(self):
        """Completely invalid response should return None, not raise."""
        result = _parse_mapping_response("This is not JSON at all!")
        assert result is None

    def test_bedrock_failure_returns_none(self):
        """call_bedrock_for_mapping should return None on failure, not raise."""
        from churn_pipeline.llm.auto_mapping import call_bedrock_for_mapping

        # Pass an invalid client that will fail
        result = call_bedrock_for_mapping("test prompt", boto3_client="not_a_client")
        assert result is None
