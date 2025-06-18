"""
This module provides a metrics evaluator class for machine learning models.
"""
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)


class MetricsEvaluator:
    """
    Evaluates classification performance using multiple metrics.
    """

    def __init__(self, y_true, y_pred, y_proba=None):
        """
        :param y_true: Ground truth labels.
        :param y_pred: Predicted labels.
        :param y_proba: Predicted probabilities (optional, for AUC).
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba

    def evaluate(self):
        """Returns metrics for prediction and ground truth values in this class."""
        cm = confusion_matrix(self.y_true, self.y_pred)
        cm_df = pd.DataFrame(
            cm,
            columns=['Predicted FAW (0)', 'Predicted Awake (1)'],
            index=['Actual FAW (0)', 'Actual Awake (1)']
        )

        results = {
            "accuracy": accuracy_score(self.y_true, self.y_pred),
            "precision": precision_score(self.y_true, self.y_pred, zero_division=0),
            "recall": recall_score(self.y_true, self.y_pred, zero_division=0),
            "f1": f1_score(self.y_true, self.y_pred, zero_division=0),
            "confusion_matrix": cm_df
        }

        if self.y_proba is not None:
            try:
                results["roc_auc"] = roc_auc_score(self.y_true, self.y_proba[:, 1])
            except Exception:
                results["roc_auc"] = None

        return results
