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

**Failures are counted against a budget.** A customer whose narrative failed is
recorded with the reason, and "N/A" goes in the output. A run that fails more
customers than its failure budget raises NarrativeGenerationError and ships
nothing. Clients keep their SHAP reasons in top_3_reasons either way.
"""

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

from churn_pipeline.llm.bedrock import BedrockCallError, call_claude

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
    Output for one customer: the generated narrative, or a failure with its reason.

    Attributes:
        customer_id: Who this is about.
        narrative: Plain-English explanation (under 150 words), or "N/A" on failure.
        success: Whether a narrative came back for this customer.
        failure_reason: Why it did not, when success is False. This is what turns
            a sea of "N/A" into something an operator can act on.
    """

    customer_id: str
    narrative: str
    success: bool
    failure_reason: Optional[str] = None


class NarrativeGenerationError(RuntimeError):
    """
    Raised when a narrative run fails more customers than its budget allows.

    One missing narrative is a recorded reason in the results. Most of them
    missing is an outage, and an outage should end in an error, because a
    delivered file that says "N/A" hundreds of times is how a client finds out
    before you do. The partial results, reasons included, are on .results.
    """

    def __init__(self, message: str, results: Dict[str, "NarrativeResult"]):
        super().__init__(message)
        self.results = results


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

    # SYSTEM_PROMPT is deliberately absent here: call_bedrock_for_narratives passes
    # it in the system channel. Including it in the user message too would send the
    # same instructions twice, and bill for them twice.
    prompt = f"""{features_section}For each customer below, write a plain-English paragraph (under 150 words) explaining why they are at risk of leaving. Reference the specific factors listed.

Format your response as:
CUSTOMER_ID: [customer_id]
NARRATIVE: [your explanation]

(Repeat for each customer)

{customers_section}"""

    return prompt


def call_bedrock_for_narratives(
    prompt: str,
    boto3_client=None,
    model_id: Optional[str] = None,
) -> Dict[str, str]:
    """
    Call Amazon Bedrock (Claude) to generate narratives for a batch.

    Args:
        prompt: The narrative prompt (from build_narrative_prompt).
        boto3_client: Optional pre-configured Bedrock client (for testing).
        model_id: Which Claude model to use. Defaults to the model for
            whichever region the client resolved to.

    Returns:
        Dict mapping customer_id to narrative text.

    Raises:
        BedrockCallError: if the call fails, straight from call_claude.
    """
    response_text = call_claude(
        prompt,
        system=SYSTEM_PROMPT,
        max_tokens=8192,
        boto3_client=boto3_client,
        model_id=model_id,
    )
    return parse_narrative_response(response_text, [])


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
        expected_customer_ids: Customer IDs this reply should contain. Any that are
            missing get a logged warning. Pass an empty list to skip the check.

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

    # A batch reply can come back short: the model skips someone, or the response was
    # truncated. Silence there turns into "N/A" further up with no reason attached, so
    # say which ones went missing while we still know.
    for customer_id in expected_customer_ids:
        if customer_id not in narratives:
            logger.warning(f"No narrative came back for {customer_id}")

    return narratives


def generate_narratives_for_batch(
    scored_customers: List[NarrativeRequest],
    batch_size: int = 50,
    feature_definitions: Dict[str, str] = None,
    boto3_client=None,
    failure_budget: float = 0.10,
) -> Dict[str, NarrativeResult]:
    """
    Generate narratives for every customer, in batches, and account for failures.

    Each batch is one Bedrock call. When a call fails, every customer in that
    batch is recorded with the reason and "N/A" as the narrative; when a reply
    skips a customer, that is recorded too. The run then compares its failure
    rate against failure_budget:

    - At or under the budget, the results come back with the failed customers
      in them, so the caller can put the rate and the reasons in its run summary.
    - Over the budget, the run raises NarrativeGenerationError. Shipping a file
      that is mostly "N/A" is worse than shipping nothing, and the partial
      results ride on the exception's .results for whoever handles it.

    Args:
        scored_customers: All customers needing narratives.
        batch_size: How many customers per Bedrock call (default: 50).
        feature_definitions: Optional feature descriptions for better narratives.
        boto3_client: Optional pre-configured Bedrock client.
        failure_budget: Highest tolerable fraction of failed customers
            (default: 0.10).

    Returns:
        Dict mapping customer_id to NarrativeResult.

    Raises:
        NarrativeGenerationError: when the failure rate exceeds failure_budget.
    """
    results: Dict[str, NarrativeResult] = {}

    for i in range(0, len(scored_customers), batch_size):
        batch = scored_customers[i : i + batch_size]
        prompt = build_narrative_prompt(batch, feature_definitions)

        try:
            narratives = call_bedrock_for_narratives(prompt, boto3_client=boto3_client)
        except BedrockCallError as e:
            # The whole batch failed for one reason; record it on every customer in it.
            for req in batch:
                results[req.customer_id] = NarrativeResult(
                    customer_id=req.customer_id,
                    narrative="N/A",
                    success=False,
                    failure_reason=str(e),
                )
            continue

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
                    failure_reason="the model's reply did not include this customer",
                )

    failed = [r for r in results.values() if not r.success]
    failure_rate = len(failed) / len(results) if results else 0.0

    if failure_rate > failure_budget:
        reasons = Counter(r.failure_reason for r in failed)
        summary = "; ".join(f"{count}x {reason}" for reason, count in reasons.most_common(3))
        raise NarrativeGenerationError(
            f"{len(failed)} of {len(results)} narratives failed "
            f"({failure_rate:.0%}), over the {failure_budget:.0%} failure budget. "
            f"Reasons: {summary}",
            results=results,
        )

    return results
