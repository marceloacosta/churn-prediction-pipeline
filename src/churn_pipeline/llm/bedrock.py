"""
Talking to Bedrock — the one place that knows how the model gets called
=======================================================================

Both LLM steps in this pipeline do the same thing underneath: send a prompt,
read back text. Neither of them cares that the text comes from Claude on Amazon
Bedrock, so that detail lives here instead of being copy-pasted into both.

**Credentials are not passed around.** boto3 reads `AWS_BEARER_TOKEN_BEDROCK`
and `AWS_DEFAULT_REGION` from the environment on its own, so no function in this
codebase ever takes a key as an argument. See the "AWS credentials" setup page
for how those get set in Colab.

**Nothing here raises.** A failed call logs a warning and returns None. The
caller decides what to do without a result — the pipeline treats both LLM steps
as enhancements, not dependencies.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# The model family we use. Bedrock reaches Claude through a cross-region
# inference profile, so what actually gets sent is prefixed with the region's
# area: us-east-1 -> "us.anthropic...", eu-west-1 -> "eu.anthropic...".
DEFAULT_MODEL = "anthropic.claude-sonnet-4-6"


def resolve_model_id(region: str, base: str = DEFAULT_MODEL) -> str:
    """
    Build the inference profile ID for a region.

    A model ID that works in us-east-1 will not work in eu-west-1, and the error
    Bedrock returns for the mismatch (ValidationException) does not say so. This
    derives the prefix instead of making the caller remember it.

    Args:
        region: An AWS region name, e.g. "us-east-1".
        base: The model to reach. Defaults to DEFAULT_MODEL.

    Returns:
        The prefixed model ID, e.g. "us.anthropic.claude-sonnet-4-6".
    """
    return f"{region.split('-')[0]}.{base}"


def call_claude(
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 4096,
    boto3_client=None,
    model_id: Optional[str] = None,
) -> Optional[str]:
    """
    Send one prompt to Claude on Bedrock and return the reply text.

    Uses the Converse API, which takes the same shape for every model on Bedrock
    — no provider-specific request body to hand-assemble.

    Args:
        prompt: The user message.
        system: Optional system prompt.
        max_tokens: Ceiling on the reply length.
        boto3_client: Optional pre-configured bedrock-runtime client (for testing).
        model_id: Override the model. Defaults to DEFAULT_MODEL, prefixed for
            whichever region the client resolved to.

    Returns:
        The reply text, or None if the call failed for any reason — no
        credentials, no region, model not enabled for this account, network
        failure, unexpected response shape. The logged warning names which.
    """
    import boto3

    try:
        if boto3_client is None:
            boto3_client = boto3.client("bedrock-runtime")

        if model_id is None:
            region = getattr(getattr(boto3_client, "meta", None), "region_name", None)
            model_id = resolve_model_id(region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))

        kwargs = {
            "modelId": model_id,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": max_tokens},
        }
        if system:
            kwargs["system"] = [{"text": system}]

        response = boto3_client.converse(**kwargs)
        return response["output"]["message"]["content"][0]["text"]

    except Exception as e:
        logger.warning(f"Bedrock call failed: {type(e).__name__}: {e}")
        return None
