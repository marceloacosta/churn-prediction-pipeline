"""
LLM Narrative Generator — Turning SHAP Numbers Into Plain English
==================================================================

SHAP gives us "contract_type=month-to-month (+0.23)". That's useful for a data
scientist, but meaningless to a business user who needs to decide whether to call
this customer today.

A human would read that SHAP output and write: "This customer has no long-term
commitment. Month-to-month customers are 3x more likely to leave than those on
annual contracts. Combined with their high support ticket count, this customer
hasn't built loyalty yet."

The LLM does this translation at scale — for dozens of customers in seconds.

**Cost efficiency:** We batch customers (default: 50 per prompt) into a single
Bedrock call. At ~$0.003 per 1K input tokens, processing 50 customers costs
roughly $0.01-0.02 per batch. The whole step costs cents, not dollars.

**Non-blocking:** If Bedrock fails, the narrative_explanation field is set to
"N/A" and the pipeline continues. Clients still get SHAP reasons in the
top_3_reasons column — just not the English paragraphs.
"""

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert customer retention analyst writing for non-technical \
business users. For each customer, explain in plain English WHY the model predicts they \
are at risk of leaving. Reference specific feature values. Use simple language. \
Keep each explanation under 150 words. Do not use technical jargon like 'SHAP values' \
or 'feature importance.'"""


@dataclass
class NarrativeRequest:
    """
    Input for narrative generation — one customer's prediction context.

    Attributes:
        customer_id: Who is this about?
        churn_probability: How likely they are to leave (0.0 to 1.0).
        risk_tier: Human-friendly label (high/medium/low).
        top_shap_features: List of dicts with {feature, value, contribution}.
    """

    customer_id: str
    churn_probability: float
    risk_tier: str
    top_shap_features: List[Dict]


@dataclass
class NarrativeResult:
    """
    Output for one customer — the generated narrative or a failure marker.

    Attributes:
        customer_id: Who this is about.
        narrative: Plain-English explanation (under 150 words), or "N/A" on failure.
        success: Whether the LLM call succeeded for this customer.
    """

    customer_id: str
    narrative: str
    success: bool


def build_narrative_prompt(
    batch: List[NarrativeRequest],
    feature_definitions: Dict[str, str] = None,
) -> str:
    """
    Construct a prompt containing multiple customers for batch processing.

    Instead of making 50 separate API calls (expensive, slow), we pack all
    customers into a single prompt. The LLM writes one paragraph per customer
    in a single response.

    The prompt includes:
    - Feature definitions (so the LLM understands what each feature means)
    - Each customer's ID, probability, tier, and SHAP features
    - Instructions for format and style

    Args:
        batch: List of NarrativeRequest objects (one per customer).
        feature_definitions: Optional dict mapping feature names to plain-English
            descriptions. Helps the LLM write better narratives.

    Returns:
        The complete prompt string ready to send to Bedrock.
    """
    # Build feature definitions section
    features_section = ""
    if feature_definitions:
        features_section = "Feature definitions (for your reference):\n"
        for name, description in feature_definitions.items():
            features_section += f"  - {name}: {description}\n"
        features_section += "\n"

    # Build customer sections
    customers_section = ""
    for req in batch:
        customers_section += f"\n--- Customer: {req.customer_id} ---\n"
        customers_section += f"Churn probability: {req.churn_probability:.2f}\n"
        customers_section += f"Risk tier: {req.risk_tier}\n"
        customers_section += "Top contributing factors:\n"
        for feat in req.top_shap_features:
            feature_name = feat.get("feature", "unknown")
            contribution = feat.get("contribution", 0.0)
            sign = "+" if contribution >= 0 else ""
            customers_section += f"  - {feature_name} ({sign}{contribution:.2f})\n"

    prompt = f"""{SYSTEM_PROMPT}

{features_section}For each customer below, write a plain-English paragraph (under 150 words) explaining why they are at risk of leaving. Reference the specific factors listed.

Format your response as:
CUSTOMER_ID: [customer_id]
NARRATIVE: [your explanation]

(Repeat for each customer)

{customers_section}"""

    return prompt


def call_bedrock_for_narratives(
    prompt: str,
    boto3_client=None,
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
) -> Optional[Dict[str, str]]:
    """
    Call Amazon Bedrock (Claude) to generate narratives for a batch.

    Returns {customer_id: narrative_text} or None on failure.
    Non-blocking: if Bedrock fails, returns None.

    Args:
        prompt: The narrative prompt (from build_narrative_prompt).
        boto3_client: Optional pre-configured Bedrock client (for testing).
        model_id: Which Claude model to use.

    Returns:
        Dict mapping customer_id to narrative text, or None if the call failed.
    """
    import boto3

    try:
        if boto3_client is None:
            boto3_client = boto3.client("bedrock-runtime")

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8192,
            "system": SYSTEM_PROMPT,
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

        return parse_narrative_response(response_text, [])

    except Exception as e:
        logger.warning(f"Bedrock narrative generation failed: {e}")
        return None


def parse_narrative_response(
    response_text: str,
    expected_customer_ids: List[str],
) -> Dict[str, str]:
    """
    Parse the LLM's response into individual customer narratives.

    Expects format:
    CUSTOMER_ID: CUST_001
    NARRATIVE: This customer is at risk because...

    Args:
        response_text: Raw text from the LLM response.
        expected_customer_ids: List of customer IDs we expect narratives for.

    Returns:
        Dict mapping customer_id to narrative text.
    """
    narratives: Dict[str, str] = {}
    current_id = None
    current_narrative_lines: List[str] = []

    for line in response_text.split("\n"):
        line_stripped = line.strip()

        if line_stripped.startswith("CUSTOMER_ID:"):
            # Save previous customer if exists
            if current_id is not None:
                narratives[current_id] = " ".join(current_narrative_lines).strip()

            current_id = line_stripped.replace("CUSTOMER_ID:", "").strip()
            current_narrative_lines = []

        elif line_stripped.startswith("NARRATIVE:"):
            narrative_text = line_stripped.replace("NARRATIVE:", "").strip()
            if narrative_text:
                current_narrative_lines.append(narrative_text)

        elif current_id is not None and line_stripped:
            # Continuation of a narrative
            current_narrative_lines.append(line_stripped)

    # Save the last customer
    if current_id is not None:
        narratives[current_id] = " ".join(current_narrative_lines).strip()

    return narratives


def generate_narratives_for_batch(
    scored_customers: List[NarrativeRequest],
    batch_size: int = 50,
    feature_definitions: Dict[str, str] = None,
    boto3_client=None,
) -> Dict[str, NarrativeResult]:
    """
    Process all customers in batches, generating narratives for each.

    If a batch fails, those customers get success=False and narrative="N/A".
    The pipeline continues regardless — narratives are nice-to-have, not blocking.

    Args:
        scored_customers: All customers needing narratives.
        batch_size: How many customers per Bedrock call (default: 50).
        feature_definitions: Optional feature descriptions for better narratives.
        boto3_client: Optional pre-configured Bedrock client.

    Returns:
        Dict mapping customer_id to NarrativeResult.
    """
    results: Dict[str, NarrativeResult] = {}

    # Process in batches
    for i in range(0, len(scored_customers), batch_size):
        batch = scored_customers[i : i + batch_size]
        batch_ids = [req.customer_id for req in batch]

        prompt = build_narrative_prompt(batch, feature_definitions)
        narratives = call_bedrock_for_narratives(prompt, boto3_client=boto3_client)

        if narratives is None:
            # Batch failed — mark all customers in this batch as failed
            for req in batch:
                results[req.customer_id] = NarrativeResult(
                    customer_id=req.customer_id,
                    narrative="N/A",
                    success=False,
                )
        else:
            # Match narratives to customers
            for req in batch:
                if req.customer_id in narratives:
                    results[req.customer_id] = NarrativeResult(
                        customer_id=req.customer_id,
                        narrative=narratives[req.customer_id],
                        success=True,
                    )
                else:
                    results[req.customer_id] = NarrativeResult(
                        customer_id=req.customer_id,
                        narrative="N/A",
                        success=False,
                    )

    return results
