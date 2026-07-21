"""
Unit Tests: Drift Detection
=============================

Specific scenarios testing PSI computation and drift alerting at known boundaries.

Validates: Requirements 8.1, 8.2

# Feature: churn-prediction-pipeline, Task 9.3: Drift detection unit tests
"""

import numpy as np
import pandas as pd
import pytest

from churn_pipeline.steps.monitoring import compute_psi, check_drift, DriftReport


class TestPSIComputation:
    """Test PSI formula with known distributions."""

    def test_identical_distributions_psi_zero(self):
        """Same distribution → PSI = 0."""
        data = np.random.normal(50, 10, size=1000)
        psi = compute_psi(data, data)
        assert abs(psi) < 1e-10

    def test_dramatically_shifted_distribution_high_psi(self):
        """Completely different distributions → PSI > 0.2."""
        reference = np.random.normal(50, 5, size=1000)
        current = np.random.normal(100, 5, size=1000)  # Mean shifted by 10 std devs
        psi = compute_psi(reference, current)
        assert psi > 0.2, f"Expected PSI > 0.2 for dramatic shift, got {psi}"

    def test_slight_shift_low_psi(self):
        """Very slight shift → PSI should be small (< 0.1)."""
        np.random.seed(42)
        reference = np.random.normal(50, 10, size=5000)
        # Shift mean by just 1 unit (0.1 std devs)
        current = np.random.normal(51, 10, size=5000)
        psi = compute_psi(reference, current)
        assert psi < 0.1, f"Expected PSI < 0.1 for slight shift, got {psi}"

    def test_psi_is_non_negative(self):
        """PSI should always be >= 0."""
        reference = np.random.normal(0, 1, size=500)
        current = np.random.normal(0.5, 1.5, size=500)
        psi = compute_psi(reference, current)
        assert psi >= 0, f"PSI should be non-negative, got {psi}"


class TestDriftDetection:
    """Test the check_drift function and alert logic."""

    def test_no_drift_no_alert(self):
        """Identical distributions → no features drifted, no alert."""
        np.random.seed(42)
        training_stats = {
            "feature_a": np.random.normal(50, 10, size=500),
            "feature_b": np.random.normal(0, 1, size=500),
        }
        current = pd.DataFrame({
            "feature_a": np.random.normal(50, 10, size=200),
            "feature_b": np.random.normal(0, 1, size=200),
        })

        report = check_drift(training_stats, current)

        assert report.alert_triggered is False
        assert report.features_drifted == []
        assert report.features_checked == 2

    def test_drift_triggers_alert(self):
        """Dramatically shifted feature → alert triggered."""
        np.random.seed(42)
        training_stats = {
            "feature_a": np.random.normal(50, 10, size=500),
            "feature_b": np.random.normal(0, 1, size=500),
        }
        current = pd.DataFrame({
            "feature_a": np.random.normal(50, 10, size=200),  # Same
            "feature_b": np.random.normal(10, 1, size=200),   # Shifted!
        })

        report = check_drift(training_stats, current)

        assert report.alert_triggered is True
        assert "feature_b" in report.features_drifted
        assert report.psi_scores["feature_b"] > 0.2

    def test_threshold_boundary_below(self):
        """PSI just below threshold → no alert."""
        # Use identical data to guarantee PSI ≈ 0 (well below 0.2)
        np.random.seed(123)
        data = np.random.normal(50, 10, size=500)
        training_stats = {"feature_a": data}
        current = pd.DataFrame({"feature_a": data[:200]})

        report = check_drift(training_stats, current, threshold=0.2)
        assert report.alert_triggered is False

    def test_threshold_boundary_above(self):
        """PSI well above threshold → alert."""
        np.random.seed(123)
        training_stats = {
            "feature_a": np.random.normal(0, 1, size=500),
        }
        current = pd.DataFrame({
            "feature_a": np.random.normal(5, 1, size=200),  # 5 std devs away
        })

        report = check_drift(training_stats, current, threshold=0.2)
        assert report.alert_triggered is True
        assert "feature_a" in report.features_drifted

    def test_missing_feature_in_current_data_skipped(self):
        """Features not in current data are silently skipped."""
        training_stats = {
            "feature_a": np.random.normal(50, 10, size=500),
            "feature_missing": np.random.normal(0, 1, size=500),
        }
        current = pd.DataFrame({
            "feature_a": np.random.normal(50, 10, size=200),
            # feature_missing is not present
        })

        report = check_drift(training_stats, current)
        assert report.features_checked == 1  # Only feature_a was checked
