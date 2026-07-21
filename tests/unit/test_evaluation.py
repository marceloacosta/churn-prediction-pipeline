"""
Unit Tests: Evaluation Gate Edge Cases
========================================

Tests specific boundary conditions and model card structure.

Validates: Requirements 5.1, 5.2, 5.4

# Feature: churn-prediction-pipeline, Task 10.3: Evaluation edge cases
"""

import numpy as np
import pytest

from churn_pipeline.steps.evaluation import evaluate_model, generate_model_card


class TestEvaluationGateBoundary:
    """Test pass/fail at exact boundary values."""

    def _make_data_with_auc(self, target_auc: float, n: int = 200):
        """Helper to create data that achieves approximately the target AUC."""
        n_pos = n // 5  # 20% positive rate
        n_neg = n - n_pos
        y_true = np.array([0] * n_neg + [1] * n_pos)

        # To control AUC precisely, we create predictions where
        # positives get high probs and negatives get low probs,
        # with overlap controlled by the target
        if target_auc >= 0.70:
            # Good separation
            y_pred = np.where(
                y_true == 1,
                np.random.uniform(0.6, 1.0, size=n),
                np.random.uniform(0.0, 0.5, size=n),
            )
        else:
            # Poor separation (near random)
            y_pred = np.random.uniform(0.3, 0.7, size=n)

        return y_true, y_pred

    def test_auc_exactly_070_passes(self):
        """AUC of exactly 0.70 should pass (>= not >)."""
        # Create perfectly controlled data
        # 10 positives, 10 negatives, arranged so AUC = exactly 0.70
        # AUC = P(score(pos) > score(neg)) for random pairs
        np.random.seed(42)
        y_true = np.array([0] * 100 + [1] * 100)
        # Positives all get 0.8, negatives: 70% get 0.3, 30% get 0.9
        # This gives AUC = 0.70 (positives beat 70% of negatives)
        y_pred = np.zeros(200)
        y_pred[100:] = 0.8  # All positives at 0.8
        y_pred[:70] = 0.3   # 70 negatives below positives
        y_pred[70:100] = 0.9  # 30 negatives above positives

        result = evaluate_model(y_true, y_pred, min_auc_threshold=0.70)
        assert result.auc_roc == 0.70
        assert result.passed is True

    def test_auc_0699_fails(self):
        """AUC of 0.699 should fail (below threshold)."""
        # Similar setup but with slightly less separation
        y_true = np.array([0] * 1000 + [1] * 1000)
        y_pred = np.zeros(2000)
        y_pred[1000:] = 0.8  # All positives at 0.8
        y_pred[:699] = 0.3   # 699 negatives below
        y_pred[699:1000] = 0.9  # 301 negatives above

        result = evaluate_model(y_true, y_pred, min_auc_threshold=0.70)
        assert result.auc_roc < 0.70
        assert result.passed is False

    def test_gate_ignores_f1(self):
        """Pass/fail depends only on AUC, not F1."""
        # High AUC but potentially low F1 (good ranking, bad threshold)
        np.random.seed(42)
        y_true = np.array([0] * 80 + [1] * 20)
        # Good ranking (high AUC) but all predictions above 0.5 → bad precision
        y_pred = np.where(
            y_true == 1,
            np.random.uniform(0.85, 0.95, size=100),
            np.random.uniform(0.55, 0.75, size=100),
        )

        result = evaluate_model(y_true, y_pred, min_auc_threshold=0.70)
        # AUC should be high (good ranking)
        assert result.auc_roc >= 0.70
        assert result.passed is True
        # F1 might be low because threshold=0.5 classifies many negatives as positive


class TestModelCard:
    """Test model card generation structure."""

    def test_model_card_contains_all_required_keys(self):
        """Model card must have all required metadata fields."""
        eval_result = evaluate_model(
            np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0]),
            np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.2, 0.4, 0.6, 0.1]),
        )
        card = generate_model_card(
            eval_result=eval_result,
            training_params={"max_depth": 6, "eta": 0.1},
            dataset_info={"client_id": "telco_ibm", "rows": 7043, "training_date": "2024-01-15"},
            feature_list=["tenure_months", "monthly_charges", "total_charges"],
            model_id="churn-xgb-v3",
        )

        # Required top-level keys
        assert "model_id" in card
        assert "training_date" in card
        assert "client_id" in card
        assert "dataset_rows" in card
        assert "feature_count" in card
        assert "features_used" in card
        assert "metrics" in card
        assert "hyperparameters" in card
        assert "evaluation_gate" in card
        assert "data_contract_version" in card

    def test_model_card_metrics_structure(self):
        """Metrics section must contain auc_roc, f1, precision, recall."""
        eval_result = evaluate_model(
            np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0]),
            np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.2, 0.4, 0.6, 0.1]),
        )
        card = generate_model_card(
            eval_result=eval_result,
            training_params={},
            dataset_info={},
            feature_list=["f1", "f2"],
        )

        metrics = card["metrics"]
        assert "auc_roc" in metrics
        assert "f1_score" in metrics
        assert "precision" in metrics
        assert "recall" in metrics

    def test_model_card_gate_section(self):
        """Evaluation gate section includes passed, threshold, margin."""
        eval_result = evaluate_model(
            np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0]),
            np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.2, 0.4, 0.6, 0.1]),
        )
        card = generate_model_card(
            eval_result=eval_result,
            training_params={},
            dataset_info={},
            feature_list=["f1"],
        )

        gate = card["evaluation_gate"]
        assert "passed" in gate
        assert "threshold" in gate
        assert "margin" in gate
        assert gate["threshold"] == 0.70
