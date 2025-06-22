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

    def __init__(self, class_0, class_1, y_true, y_pred, y_proba=None):
        """
        Evaluates classification performance using multiple metrics.
        :param class_0: Class 0 name
        :param class_1: Class 1 name
        :param y_true: Ground truth labels.
        :param y_pred: Predicted labels.
        :param y_proba: Predicted probabilities (optional, for AUC).
        """
        self.class_0 = class_0
        self.class_1 = class_1
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba

    def evaluate(self, print_res=True):
        """
        Returns metrics for prediction and ground truth values in this class.
        :param print_res: If true, prints results to console.
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        pred_class_0_name = f"Predicted {self.class_0} (0)"
        pred_class_1_name = f"Predicted {self.class_1} (1)"
        true_class_0_name = f"True {self.class_0} (0)"
        true_class_1_name = f"True {self.class_1} (1)"
        cm_df = pd.DataFrame(
            cm,
            columns=[pred_class_0_name, pred_class_1_name],
            index=[true_class_0_name, true_class_1_name]
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

            # Make
        if print_res:
            print(f"Accuracy: {results['accuracy']}")
            print(f"Precision: {results['precision']}")
            print(f"Recall: {results['recall']}")
            print(f"F1 Score: {results['f1']}")
            print(f"ROC AUC: {results['roc_auc']}")
            print(f"Confusion Matrix:")
            print(f"{results['confusion_matrix']}")

        return results
