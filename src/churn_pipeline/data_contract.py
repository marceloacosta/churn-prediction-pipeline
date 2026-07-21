"""
Data Contract — The Universal Form Every Dataset Must Fill Out
=============================================================

Imagine you run a hospital. Every patient who walks in fills out the same intake form.
Some fields are mandatory (name, date of birth, reason for visit) — without these,
you literally can't treat them. Other fields are helpful (allergies, insurance info)
but you can still proceed without them.

Our data contract works the same way. Every client's customer data must conform to
a standard "form" with three tiers:

- **Tier 1 (Required):** The pipeline cannot function without these. If customer_id
  is missing, we don't know WHO we're predicting about. If churn_label is missing,
  we have nothing to learn from. These are non-negotiable.

- **Tier 2 (Engagement):** These make the model significantly better — contract type,
  payment method, support tickets. A model without these is like a doctor diagnosing
  without asking about symptoms. It can still make a guess, just a worse one.

- **Tier 3 (Demographics):** Nice-to-have context. Gender, age, partner status.
  These add nuance but the model works fine without them.

The key insight: Tier 1 is a hard gate (missing = rejected), while Tier 2/3 are
soft signals (missing = logged but accepted). This lets us onboard clients with
varying data richness without blocking anyone who has the essentials.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Tier(Enum):
    """
    Data field importance levels — think of them as priority lanes.

    REQUIRED (1): No entry without these. The bouncer checks for these at the door.
    ENGAGEMENT (2): The VIP upgrades — makes everything better but not mandatory.
    DEMOGRAPHIC (3): The optional extras — nice context, never a dealbreaker.
    """

    REQUIRED = 1
    ENGAGEMENT = 2
    DEMOGRAPHIC = 3


@dataclass(frozen=True)
class FieldSpec:
    """
    A single field in our data contract — one line on the intake form.

    Think of it as a slot with a label, a type expectation, a priority level,
    and a human-readable explanation of what goes there.

    Attributes:
        name: The standardized field name (what we call it internally).
        dtype: Expected data type — "float", "int", "category", or "string".
        tier: How critical this field is (Tier 1/2/3).
        description: Plain-English explanation of what this field represents.
        allowed_values: For categorical fields, the valid options. None means any value.
    """

    name: str
    dtype: str
    tier: Tier
    description: str
    allowed_values: Optional[List[str]] = field(default=None)


# ---------------------------------------------------------------------------
# The Standard Schema — our universal intake form
# ---------------------------------------------------------------------------
# Every field the pipeline knows about, organized by tier. A client's dataset
# doesn't need ALL of these — just all of Tier 1. The rest is gravy.

STANDARD_SCHEMA: Dict[str, FieldSpec] = {
    # --- Tier 1: Required (the non-negotiables) ---
    "customer_id": FieldSpec(
        name="customer_id",
        dtype="string",
        tier=Tier.REQUIRED,
        description="Unique customer identifier — who are we talking about?",
    ),
    "tenure_months": FieldSpec(
        name="tenure_months",
        dtype="int",
        tier=Tier.REQUIRED,
        description="How many months this person has been a customer. "
        "Longer tenure generally means lower churn risk.",
    ),
    "monthly_charges": FieldSpec(
        name="monthly_charges",
        dtype="float",
        tier=Tier.REQUIRED,
        description="What the customer pays each month. "
        "Higher charges can signal both value and price sensitivity.",
    ),
    "total_charges": FieldSpec(
        name="total_charges",
        dtype="float",
        tier=Tier.REQUIRED,
        description="Cumulative amount billed over the customer's lifetime. "
        "Roughly tenure × monthly_charges, but accounts for plan changes.",
    ),
    "churn_label": FieldSpec(
        name="churn_label",
        dtype="int",
        tier=Tier.REQUIRED,
        description="The answer we're trying to predict: 1 = customer left, "
        "0 = customer stayed. This is what the model learns from.",
    ),
    # --- Tier 2: Engagement (the helpful extras) ---
    "contract_type": FieldSpec(
        name="contract_type",
        dtype="category",
        tier=Tier.ENGAGEMENT,
        description="Contract duration — month-to-month customers leave 3x more "
        "often than those locked into annual contracts.",
        allowed_values=["month-to-month", "one_year", "two_year"],
    ),
    "payment_method": FieldSpec(
        name="payment_method",
        dtype="category",
        tier=Tier.ENGAGEMENT,
        description="How the customer pays. Auto-pay customers tend to be stickier "
        "(less friction to stay).",
    ),
    "support_tickets": FieldSpec(
        name="support_tickets",
        dtype="int",
        tier=Tier.ENGAGEMENT,
        description="Number of support contacts. High ticket counts often signal "
        "frustration — a leading indicator of churn.",
    ),
    # --- Tier 3: Demographics (the nice-to-haves) ---
    "gender": FieldSpec(
        name="gender",
        dtype="category",
        tier=Tier.DEMOGRAPHIC,
        description="Customer gender. Rarely a strong churn predictor on its own, "
        "but can interact with other features.",
    ),
    "age_bucket": FieldSpec(
        name="age_bucket",
        dtype="category",
        tier=Tier.DEMOGRAPHIC,
        description="Age range bucket (e.g., 18-25, 26-35). Different age groups "
        "may have different switching costs and loyalty patterns.",
    ),
    "partner_status": FieldSpec(
        name="partner_status",
        dtype="category",
        tier=Tier.DEMOGRAPHIC,
        description="Whether the customer has a partner. Households with shared "
        "accounts tend to have higher switching costs.",
    ),
}


def get_fields_by_tier(tier: Tier) -> List[FieldSpec]:
    """Return all fields belonging to a specific tier."""
    return [spec for spec in STANDARD_SCHEMA.values() if spec.tier == tier]


def get_tier1_field_names() -> List[str]:
    """Return just the names of Tier 1 (required) fields — the bouncer's checklist."""
    return [name for name, spec in STANDARD_SCHEMA.items() if spec.tier == Tier.REQUIRED]
