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
            return self._evaluate_single("single", print_res)

        all_results = []
        for i in range(len(self.y_true)):
            y_true_i = self.y_true[i]
            y_pred_i = self.y_pred[i]
            y_proba_i = self.y_proba[i] if self.y_proba is not None else None

            evaluator = MetricsEvaluator(self.class_0, self.class_1, y_true_i, y_pred_i, y_proba_i)
            result = evaluator._evaluate_single(f"fold_{i+1}", print_res)
            all_results.append(result)

        summary = self._calculate_summary(all_results)

        # Prints a nice summary confusion matrix
        if print_res:
            self.print_result(summary, True)

        return {"individual_results": all_results, "summary": summary}

    def _evaluate_single(self, result_name: str, print_res=True):
        """
        Evaluates a single set of predictions.
        :param result_name: Name of the result for the correct assignment of metrics to result.
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

        raw_results = {
            "result": result_name,
            "accuracy": accuracy_score(self.y_true, self.y_pred),
            "precision": precision_score(self.y_true, self.y_pred, zero_division=0),
            "recall": recall_score(self.y_true, self.y_pred, zero_division=0),
            "f1": f1_score(self.y_true, self.y_pred, zero_division=0),
            "confusion_matrix": cm_df
        }

        if self.y_proba is not None:
            try:
                raw_results["roc_auc"] = roc_auc_score(self.y_true, self.y_proba[:, 1])
            except Exception:
                raw_results["roc_auc"] = None

        # Round results
        results = {
            k: round(v, 4) if isinstance(v, (float, int)) and k != "confusion_matrix" else v
            for k, v in raw_results.items()
        }

        if print_res:
            self.print_result(results, False)

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
        std_matrix = np.std(relative_matrices, axis=0)

        template_df = results_list[0]["confusion_matrix"]
        mean_df = pd.DataFrame(mean_matrix, columns=template_df.columns, index=template_df.index)
        std_df = pd.DataFrame(std_matrix, columns=template_df.columns, index=template_df.index)

        return mean_df, std_df

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
                summary[metric] = {"mean": np.mean(values), "standard_deviation": np.std(values)}
            else:
                summary[metric] = {"mean": None, "standard_deviation": None}

        # Add confusion matrix summary
        mean_cm, std_cm = MetricsEvaluator._calculate_confusion_matrix_summary(results_list)
        summary["confusion_matrix"] = {"mean": mean_cm, "standard_deviation": std_cm}

        return summary

    def print_result(self, result: dict, summary: bool):
        if summary:
            self._print_summary(result)
        else:
            self._print_single(result)

    @staticmethod
    def _print_summary(summary_dict: dict):
        """
        Prints the summary statistics given in the input dictionary.

        The method processes the provided dictionary, which contains statistical data
        (e.g., mean and variance) for various metrics, including confusion matrices
        if available, and outputs the formatted result to the console.

        :param summary_dict: A dictionary containing statistical metrics. The keys are
            metric names (e.g., "accuracy", "precision", or "confusion_matrix"), and
            the values are sub-dictionaries with "mean" and "variance" keys. For
            "confusion_matrix", the values should be DataFrames where the mean and
            variance are stored by row and column.
        :type summary_dict: dict
        :return: None
        """
        try:
            print("Summary Statistics:")
            for metric, stats in summary_dict.items():
                if metric == "confusion_matrix":
                    print("\nConfusion Matrix (mean % ± std %):")
                    combined_matrix = stats["mean"].copy()
                    for i in stats["mean"].index:
                        for j in stats["mean"].columns:
                            combined_matrix.loc[
                                i, j] = f"{stats['mean'].loc[i, j]:.1f}% ± {stats['standard_deviation'].loc[i, j]:.1f}%"
                    print(combined_matrix)
                else:
                    print(f"{metric} - Mean: {stats['mean']*100:.1f}%, Standard deviation: {stats['standard_deviation']*100:.1f}%")
        except KeyError:
            print(f"Following Error was encountered: {KeyError} \n Trying old Format with Variance instead of Std")
            print("Summary Statistics:")
            for metric, stats in summary_dict.items():
                if metric == "confusion_matrix":
                    print("\nConfusion Matrix (mean % ± var %):")
                    combined_matrix = stats["mean"].copy()
                    for i in stats["mean"].index:
                        for j in stats["mean"].columns:
                            combined_matrix.loc[
                                i, j] = f"{stats['mean'].loc[i, j]:.1f}% ± {stats['variance'].loc[i, j]:.1f}%"
                    print(combined_matrix)
                else:
                    print(f"{metric} - Mean: {stats['mean']*100:.1f}%, Variance: {stats['variance']*100:.1f}%")



    @staticmethod
    def _print_single(result_dict: dict):
        """
        Prints the details of a single result dictionary containing evaluation
        metrics.

        It prints details including fold number, accuracy, precision, recall,
        F1 score, ROC AUC, and confusion matrix. The metrics are provided
        in the `result_dict` parameter.

        :param result_dict: A dictionary containing the following keys:
            - 'result': Fold number.
            - 'accuracy': The model's accuracy.
            - 'precision': The model's precision.
            - 'recall': The model's recall.
            - 'f1': The model's F1 score.
            - 'roc_auc': The model's ROC AUC.
            - 'confusion_matrix': The confusion matrix of the evaluation.
        :type result_dict: dict
        :return: None
        """
        print(f"Fold number: {result_dict['result']}")
        print(f"Accuracy: {result_dict['accuracy']:.4f}")
        print(f"Precision: {result_dict['precision']:.4f}")
        print(f"Recall: {result_dict['recall']:.4f}")
        print(f"F1 Score: {result_dict['f1']:.4f}")
        print(f"ROC AUC: {result_dict['roc_auc']:.4f}")
        print(f"Confusion Matrix:")
        print(f"{result_dict['confusion_matrix']}\n")