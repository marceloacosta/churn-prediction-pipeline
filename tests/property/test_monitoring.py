"""
Property 10: PSI Symmetry Baseline
====================================

For ANY numerical distribution (array of floats), computing PSI of that
distribution against itself SHALL produce a value of 0.0 (within floating-point
epsilon of 1e-10).

If something hasn't changed, the "change detector" must report zero change.
This is the sanity check for our PSI implementation — any non-zero result when
comparing identical data means the math is wrong.

Validates: Requirements 8.1

# Feature: churn-prediction-pipeline, Property 10: PSI Symmetry Baseline
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from churn_pipeline.steps.monitoring import compute_psi


@st.composite
def numeric_distribution(draw):
    """Generate a random float array with 100+ elements."""
    n = draw(st.integers(min_value=100, max_value=500))
    # Generate from a normal-ish distribution with random mean/std
    mean = draw(st.floats(min_value=-100, max_value=100))
    std = draw(st.floats(min_value=0.1, max_value=50))
    return np.random.normal(mean, std, size=n)


@pytest.mark.property
@given(data=numeric_distribution())
@settings(max_examples=200, deadline=None)
def test_psi_self_comparison_is_zero(data: np.ndarray) -> None:
    """
    PSI of any distribution compared against itself must be ~0.0.
    """
    psi = compute_psi(data, data)

    assert abs(psi) < 1e-10, (
        f"PSI of distribution against itself should be ~0.0, got {psi}. "
        f"Data: n={len(data)}, mean={data.mean():.2f}, std={data.std():.2f}"
    )
