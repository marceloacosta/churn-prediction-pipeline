"""
Evaluation Gate — The Chef's Taste Test
=========================================

Before a restaurant serves a dish, a chef tastes it. If it's not good enough,
it doesn't go to the table. The evaluation gate is that taste test — the model
must prove it can sort customers by churn risk significantly better than random
chance, or it gets rejected and never reaches the client.

The key metric is **AUC-ROC** (Area Under the Receiver Operating Characteristic
curve). Imagine sorting all customers by how likely the model thinks they'll leave.
A perfect model puts all actual leavers at the top of the list:

- **1.0** = perfect (all leavers at the top)
- **0.5** = random (useless — same as flipping a coin)
- **0.70 (our minimum)** = the model puts actual leavers meaningfully higher

We also track F1 score, which balances two competing mistakes:
- **Precision:** "Of everyone I flagged, how many actually left?" (false alarm rate)
- **Recall:** "Of everyone who left, how many did I flag?" (miss rate)

F1 is the harmonic mean of these two — it's high only when BOTH are good.

**The gate rule is simple:** pass/fail depends SOLELY on AUC-ROC >= 0.70.
Other metrics are logged for information but don't affect the gate decision.
A model with AUC 0.71 and terrible F1 still passes. A model with AUC 0.69
and perfect F1 still fails. Only AUC controls the gate.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvaluationResult:
    """
    The chef's verdict — did this model pass the quality gate?

    Attributes:
        passed: Did the model pass? True if AUC >= threshold.
        auc_roc: How well the model sorts customers (0.5 = random, 1.0 = perfect).
        f1_score: Balance between precision and recall (0.0-1.0).
        precision: Of flagged customers, what fraction actually left?
        recall: Of actual leavers, what fraction did we catch?
        threshold: The minimum AUC required to pass (default: 0.70).
    """

    passed: bool
    auc_roc: float
    f1_score: float
    precision: float
    recall: float
    threshold: float


def evaluate_model(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    min_auc_threshold: float = 0.70,
) -> EvaluationResult:
    """
    Evaluate model on held-out test set and decide: pass or fail?

    The gate decision depends SOLELY on AUC-ROC >= threshold.
    Other metrics are computed and logged but do NOT affect the decision.

    Why AUC and not accuracy? Because accuracy is misleading with imbalanced
    data. A model that predicts "won't churn" for everyone gets 85% accuracy
    when only 15% churn — but catches zero actual leavers. AUC measures
    RANKING quality: does the model put actual leavers higher in the sorted
    list than non-leavers?

    Args:
        y_true: Actual labels (0 or 1), shape (n_samples,).
        y_pred_proba: Predicted probabilities (0.0 to 1.0), shape (n_samples,).
        min_auc_threshold: Minimum AUC to pass (default: 0.70).

    Returns:
        EvaluationResult with all metrics and the pass/fail verdict.
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    # Compute AUC-ROC (the gate metric)
    auc = float(roc_auc_score(y_true, y_pred_proba))

    # Convert probabilities to binary predictions at 0.5 threshold for F1/precision/recall
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)

    # Compute additional metrics (informational only — don't affect the gate)
    f1 = float(f1_score(y_true, y_pred_binary, zero_division=0))
    precision = float(precision_score(y_true, y_pred_binary, zero_division=0))
    recall = float(recall_score(y_true, y_pred_binary, zero_division=0))

    # THE GATE DECISION: only AUC matters
    passed = auc >= min_auc_threshold

    return EvaluationResult(
        passed=passed,
        auc_roc=round(auc, 4),
        f1_score=round(f1, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        threshold=min_auc_threshold,
    )


def generate_model_card(
    eval_result: EvaluationResult,
    training_params: Dict,
    dataset_info: Dict,
    feature_list: List[str],
    model_id: Optional[str] = None,
) -> Dict:
    """
    Generate a 'nutrition label' for the model.

    A model card is all the metadata someone needs to understand what this
    model is, how it was trained, and how well it performs. Like a nutrition
    label on food — you can quickly assess if it's suitable for your needs.

    Args:
        eval_result: The evaluation metrics from the gate.
        training_params: Hyperparameters used (e.g., max_depth, eta, etc.).
        dataset_info: Info about training data (rows, date, client_id).
        feature_list: Ordered list of features the model uses.
        model_id: Optional model identifier. Auto-generated if not provided.

    Returns:
        Dict containing all model metadata, ready to serialize as JSON.
    """
    if model_id is None:
        model_id = f"churn-xgb-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    return {
        "model_id": model_id,
        "training_date": dataset_info.get("training_date", datetime.now().isoformat()),
        "client_id": dataset_info.get("client_id", "unknown"),
        "dataset_rows": dataset_info.get("rows", 0),
        "feature_count": len(feature_list),
        "features_used": feature_list,
        "metrics": {
            "auc_roc": eval_result.auc_roc,
            "f1_score": eval_result.f1_score,
            "precision": eval_result.precision,
            "recall": eval_result.recall,
        },
        "hyperparameters": training_params,
        "evaluation_gate": {
            "passed": eval_result.passed,
            "threshold": eval_result.threshold,
            "margin": round(eval_result.auc_roc - eval_result.threshold, 4),
        },
        "data_contract_version": "1.0",
    }
