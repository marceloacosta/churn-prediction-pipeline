"""
Unit Tests: LLM Narrative Generator Module
=============================================

Tests system prompt content, batch processing, response parsing, and failure handling.

Validates: Requirements 7.4, 7.5, 7.6

# Feature: churn-prediction-pipeline, Task 13.3: Narrative module unit tests
"""

import pytest

from churn_pipeline.llm.narrative_generator import (
    SYSTEM_PROMPT,
    NarrativeRequest,
    NarrativeResult,
    build_narrative_prompt,
    generate_narratives_for_batch,
    parse_narrative_response,
)


class TestSystemPrompt:
    """Verify the system prompt contains required instructions."""

    def test_system_prompt_mentions_non_technical(self):
        """System prompt must instruct non-technical language."""
        assert "non-technical" in SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_word_limit(self):
        """System prompt must specify under 150 words."""
        assert "150" in SYSTEM_PROMPT

    def test_system_prompt_forbids_jargon(self):
        """System prompt must tell LLM to avoid technical jargon."""
        assert "SHAP" in SYSTEM_PROMPT  # Explicitly says not to use this term


class TestBatchProcessing:
    """Verify batch behavior."""

    def test_batch_of_50_produces_single_prompt(self):
        """50 customers should generate ONE prompt, not 50 separate calls."""
        batch = [
            NarrativeRequest(
                customer_id=f"CUST_{i:03d}",
                churn_probability=0.8,
                risk_tier="high",
                top_shap_features=[
                    {"feature": "contract_type", "contribution": 0.23},
                ],
            )
            for i in range(50)
        ]

        prompt = build_narrative_prompt(batch)

        # All 50 customers must be in one prompt
        for i in range(50):
            assert f"CUST_{i:03d}" in prompt

        # It's a single string (not a list of prompts)
        assert isinstance(prompt, str)

    def test_empty_batch_produces_valid_prompt(self):
        """Empty batch should still produce a valid (if useless) prompt string."""
        prompt = build_narrative_prompt([])
        assert isinstance(prompt, str)
        assert len(prompt) > 0


class TestResponseParsing:
    """Verify narrative response parsing handles various formats."""

    def test_parse_well_formed_response(self):
        """Standard response format parses correctly."""
        response = """CUSTOMER_ID: CUST_001
NARRATIVE: This customer is at high risk because they have a month-to-month contract and have contacted support 5 times in just 2 months.

CUSTOMER_ID: CUST_002
NARRATIVE: Low risk. Long tenure and annual contract provide stability."""

        result = parse_narrative_response(response, ["CUST_001", "CUST_002"])

        assert "CUST_001" in result
        assert "CUST_002" in result
        assert "month-to-month" in result["CUST_001"]
        assert "Long tenure" in result["CUST_002"]

    def test_parse_multiline_narrative(self):
        """Narratives spanning multiple lines should be joined."""
        response = """CUSTOMER_ID: CUST_001
NARRATIVE: This customer shows several risk indicators.
They have been with us for only 2 months.
Their support ticket count is unusually high."""

        result = parse_narrative_response(response, ["CUST_001"])

        assert "CUST_001" in result
        # All lines should be joined into one narrative
        assert "several risk indicators" in result["CUST_001"]
        assert "2 months" in result["CUST_001"]
        assert "support ticket" in result["CUST_001"]

    def test_parse_empty_response(self):
        """Empty response returns empty dict."""
        result = parse_narrative_response("", ["CUST_001"])
        assert result == {}


class TestFailureHandling:
    """Verify graceful failure behavior."""

    def test_llm_failure_sets_narrative_na(self):
        """When Bedrock fails, narratives should be 'N/A' with success=False."""
        batch = [
            NarrativeRequest(
                customer_id="CUST_001",
                churn_probability=0.85,
                risk_tier="high",
                top_shap_features=[{"feature": "tenure", "contribution": 0.15}],
            ),
            NarrativeRequest(
                customer_id="CUST_002",
                churn_probability=0.90,
                risk_tier="high",
                top_shap_features=[{"feature": "contract", "contribution": 0.23}],
            ),
        ]

        # Pass invalid client to force failure
        results = generate_narratives_for_batch(
            batch, boto3_client="not_a_real_client"
        )

        assert results["CUST_001"].narrative == "N/A"
        assert results["CUST_001"].success is False
        assert results["CUST_002"].narrative == "N/A"
        assert results["CUST_002"].success is False

    def test_feature_definitions_included_in_prompt(self):
        """Feature definitions should appear in the prompt when provided."""
        batch = [
            NarrativeRequest(
                customer_id="CUST_001",
                churn_probability=0.75,
                risk_tier="high",
                top_shap_features=[{"feature": "tenure_months", "contribution": 0.15}],
            )
        ]

        definitions = {
            "tenure_months": "How long the customer has been with us",
            "monthly_charges": "What they pay each month",
        }

        prompt = build_narrative_prompt(batch, feature_definitions=definitions)

        assert "How long the customer has been with us" in prompt
        assert "What they pay each month" in prompt
