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
        :param y_true: Ground truth labels or list of ground truth labels.
        :param y_pred: Predicted labels or list of predicted labels.
        :param y_proba: Predicted probabilities or list of predicted probabilities (optional, for AUC).
        """
        self.class_0 = class_0
        self.class_1 = class_1
        self.multiple_results = isinstance(y_true, list)

        if self.multiple_results:
            if not (len(y_true) == len(y_pred)):
                raise ValueError("All input lists must have the same length")
            if y_proba is not None and len(y_true) != len(y_proba):
                raise ValueError("y_proba list must have the same length as other inputs")

        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba

    def evaluate(self, print_res=True):
        """
        Returns metrics for prediction and ground truth values in this class.
        For multiple results, returns list of results and summary statistics.
        :param print_res: If true, prints results to console.
        :return: Dictionary with metrics for each class, or list of dictionaries if multiple results.
        """
        if not self.multiple_results:
            return self._evaluate_single(print_res)

        all_results = []
        for i in range(len(self.y_true)):
            y_true_i = self.y_true[i]
            y_pred_i = self.y_pred[i]
            y_proba_i = self.y_proba[i] if self.y_proba is not None else None

            evaluator = MetricsEvaluator(self.class_0, self.class_1, y_true_i, y_pred_i, y_proba_i)
            result = evaluator._evaluate_single(print_res)
            all_results.append(result)

        summary = self._calculate_summary(all_results)

        if print_res:
            print("Summary Statistics:")
            for metric, stats in summary.items():
                if metric == "confusion_matrix":
                    print("\nConfusion Matrix (mean % ± variance %):")
                    combined_matrix = stats["mean"].copy()
                    for i in stats["mean"].index:
                        for j in stats["mean"].columns:
                            combined_matrix.loc[
                                i, j] = f"{stats['mean'].loc[i, j]:.1f}% ± {stats['variance'].loc[i, j]:.1f}%"
                    print(combined_matrix)
                else:
                    print(f"{metric} - Mean: {stats['mean']:.4f}, Variance: {stats['variance']:.4f}")

        return {"individual_results": all_results, "summary": summary}

    def _evaluate_single(self, print_res=True):
        """
        Evaluates a single set of predictions.
        :param print_res: If true, prints results to the console.
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

        if print_res:
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"Precision: {results['precision']:.4f}")
            print(f"Recall: {results['recall']:.4f}")
            print(f"F1 Score: {results['f1']:.4f}")
            print(f"ROC AUC: {results['roc_auc']:.4f}")
            print(f"Confusion Matrix:")
            print(f"{results['confusion_matrix']}")

        return results

    @staticmethod
    def _calculate_confusion_matrix_summary(results_list):
        """
        Calculates mean and variance for confusion matrices across multiple evaluations.
        :param results_list: List of evaluation results
        :return: Tuple of (mean_matrix, variance_matrix) as DataFrames with relative factors
        """
        import numpy as np

        def to_relative_factors(matrix):
            row_sums = matrix.sum(axis=1, keepdims=True)
            return matrix / row_sums * 100

        matrices = [r["confusion_matrix"].values for r in results_list]
        relative_matrices = [to_relative_factors(m) for m in matrices]

        mean_matrix = np.mean(relative_matrices, axis=0)
        var_matrix = np.var(relative_matrices, axis=0)

        template_df = results_list[0]["confusion_matrix"]
        mean_df = pd.DataFrame(mean_matrix, columns=template_df.columns, index=template_df.index)
        var_df = pd.DataFrame(var_matrix, columns=template_df.columns, index=template_df.index)

        return mean_df, var_df

    @staticmethod
    def _calculate_summary(results_list):
        """
        Calculates mean and variance for each metric across multiple evaluations.
        :param results_list: List of evaluation results
        :return: Dictionary with mean and variance for each metric
        """
        import numpy as np

        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        summary = {}

        for metric in metrics:
            values = [r[metric] for r in results_list if r[metric] is not None]
            if values:
                summary[metric] = {"mean": np.mean(values), "variance": np.var(values)}
            else:
                summary[metric] = {"mean": None, "variance": None}

        # Add confusion matrix summary
        mean_cm, var_cm = MetricsEvaluator._calculate_confusion_matrix_summary(results_list)
        summary["confusion_matrix"] = {"mean": mean_cm, "variance": var_cm}

        return summary
