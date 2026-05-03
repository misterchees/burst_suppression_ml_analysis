"""This Module contains the MetaFoldAnalyzer class."""
import os

import pandas as pd
import json
import glob
import matplotlib
import matplotlib.pyplot as plt

from MachineLearning.IO.load_data import LoadData
from MachineLearning.IO.save_result import SaveResult
from MachineLearning.Utils.path_manager import PathManager

# Agg as fallback for Headless-Environments (e.g. Test-Environments)
try:
    matplotlib.use('TkAgg')
except ImportError:
    matplotlib.use('Agg')

import seaborn as sns




class MetaFoldAnalyzer:
    """Class that calculates from results and metadata analysis from single folds overall statistics and trends."""
    def __init__(self, pm: PathManager, model_key: str, parameters: dict, run_name: str = None):
        """
        Initializes the MetaFoldAnalyzer instance with model and paths to calculated results and metadata analysis
        of single folds.

        :param pm: The global instance of PathManager.
        :param model_key: The key of the model to analyze.
        :param parameters: A dictionary containing the parameters of the epochs from which the results were
                           calculated.
        :param run_name: The name of the run from which the results and metadata analysis will be calculated.
        """
        self.model_name = model_key
        self.parameters = parameters
        self.run_name = run_name

        # Initialize Path Utilities
        self.pm = pm
        self.loader = LoadData(self.pm)
        self.saver = SaveResult(self.pm)

        # Set paths
        self.ml_results_path = self.pm.get_complex_ml_path(parameters, ["results", model_key], False, False, run_name)
        self.metadata_path = self.pm.get_complex_ml_path(
            parameters, ["metadata_analysis", model_key], False, False, run_name)

        # Container for analysis data
        self.fold_errors_by_group = {}
        self.fold_errors_of_label_by_group = {}
        self.fold_class_distributions = {}
        self.fold_metrics = {}

    def load_all_folds(self, group_col: str, label: int = 1):
        """
        Searches and loads various fold-related data including errors by group, errors of specific
        labels by group, class distributions by group, and fold metrics. The data is stored into
        appropriate instance-level attributes if the corresponding files are found.

        :param group_col: Group column indicating the attribute used for grouping data in error
            and distribution analysis.
        :type group_col: str
        :param label: Specific label value for which the error by group analysis is performed.
            Defaults to 1.
        :type label: int
        :return: None
        """
        # Search for all folds with labels and errors
        fold_files = glob.glob(os.path.join(self.ml_results_path, "*full_and_pred.csv"))
        for fold_file in fold_files:
            fold_lname = os.path.basename(fold_file).replace(".csv", "")  # long name of fold
            fold_sname = fold_lname.replace("full_and_pred", "")  # short name of fold

            # Load all files containing error by group analysis for folds
            err_path = os.path.join(self.metadata_path, f"{fold_lname}_error_by_{group_col}.csv")
            if os.path.exists(err_path):
                self.fold_errors_by_group[fold_sname] = pd.read_csv(err_path, index_col=0).copy()

            # Load all files containing error of label by group analysis for folds
            label_err_path = os.path.join(self.metadata_path, f"{fold_lname}_error_label_{label}_by_{group_col}.csv")
            if os.path.exists(label_err_path):
                self.fold_errors_of_label_by_group[fold_sname] = pd.read_csv(label_err_path, index_col=0).copy()

            # Load all files containing class distribution by group
            dist_path = os.path.join(self.metadata_path, f"{fold_lname}_class_dist_per_{group_col}.csv")
            if os.path.exists(dist_path):
                self.fold_class_distributions[fold_sname] = pd.read_csv(dist_path, header=[0,1], index_col=0).copy()

            # Load metrics for folds
            metrics_path = os.path.join(self.ml_results_path, "folds_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    self.fold_metrics[fold_sname] = json.load(f).copy()

    def aggregate_error_by_group(self, per_label=False):
        """
        Returns a combined dataframe of errors by group (e.g. ResultID) over all folds.

        :param per_label: If True, the errors of a single label are aggregated, else of both labels.
        :return: A dataframe with columns 'fold', 'group', and 'error_rate'.
        """
        combined = []
        if per_label:
            group_errors_dict = self.fold_errors_of_label_by_group
        else:
            group_errors_dict = self.fold_errors_by_group

        for fold_name, df in group_errors_dict.items():
            df = df.copy()
            df["fold"] = fold_name
            df["group"] = df.index
            combined.append(df)

        if not combined:
            return pd.DataFrame()

        return pd.concat(combined, ignore_index=True)

    def analyze_class_imbalance_vs_metric(self, metric: str):
        """
        Creates a dataframe with class distribution vs metric to see dependencies to the classes.

        :param metric: The metric to analyze.
        """
        rows = []
        for fold_name, dist in self.fold_class_distributions.items():
            if fold_name in self.fold_metrics:
                fold_number = fold_name.split("_")[0]
                result_name = f"fold_{fold_number}"

                # Get correct metric
                metric_val = None
                for entry in self.fold_metrics[fold_name].get("individual_results", []):
                    if entry.get("result") == result_name:
                        metric_val = entry.get(metric, None)
                        break

                rel = dist["rel"].mean().to_dict()
                rel[metric] = metric_val
                rel["fold"] = fold_name
                rows.append(rel)

        return pd.DataFrame(rows)

    def plot_foldwise_error_heatmap(self, group_col_name="ResultID", show_plt=True):
        """
        Plots a heatmap of error rate per Group (e.g. ResultID) in Fold.

        :param group_col_name: The name of the group for which the heatmap will be plotted.
        :param show_plt: A boolean indicating whether or not to show the heatmap.
        """
        agg = self.aggregate_error_by_group()
        if agg.empty:
            print("No errors found. Plot creation cancelled.")
            return None

        pivot = agg.pivot(index="fold", columns=group_col_name, values="error_rate")
        fig, ax = plt.subplots(figsize=(min(18, pivot.shape[1]*0.7), 6))
        sns.heatmap(pivot, annot=False, cmap="Reds", ax=ax)
        ax.set_title("Fehlerrate pro Fold und Gruppe")
        plt.tight_layout()

        if show_plt:
            plt.show()

        return fig

    def select_outlier_groups(self, df: pd.DataFrame = None, min_errors: int = 5, error_rate_threshold: float|str = "iqr",
                              iqr_multiplier: float = 1.5, save_res: bool = False) -> pd.DataFrame:
        """
        Detects groups (e.g. patients) with unusually high classification errors,
        based on a given threshold of "error_rate" and an additional minimum
        number of absolute errors.

        **Fixed column names**
          • ``error_rate`` – relative error (0...1)
          • ``incorrect_predictions`` – absolute error count

        :param df: DataFrame that contains at least the two fixed columns above. If None,
                it will be loaded from existing result analysis for given parameters.
        :param min_errors: Minimum absolute errors a group must have to be retained.
        :param iqr_multiplier: k in the Tukey rule (default 1.5 ⇒ “mild” outliers).
        :param error_rate_threshold: Threshold for the IQR method. If "iqr", the IQR is used.
        :param save_res: If True, the result is saved to a CSV file.
        :returns: Sub‐DataFrame with the outlier groups. An extra column
                  ``error_threshold`` is added for reference.
        """
        if df is None:
            folder_path = self.pm.get_complex_ml_path(
                self.parameters, ["metadata_analysis", self.model_name], False, False, self.run_name
            )
            df = pd.read_csv(folder_path / "Summary_analysis_agg_label_error_by_groups.csv")

        df = df.copy()  # Copy to prevent unwanted effects

        # Calculate threshold with iqr or set it directly depending on given value
        if error_rate_threshold == "iqr":
            threshold = self._compute_iqr_threshold(df, iqr_multiplier)
        else:
            if not isinstance(error_rate_threshold, float):
                raise ValueError("error_rate_threshold must be a float or 'iqr'.")
            threshold = error_rate_threshold

        # filter by given thresholds
        mask = (df["error_rate"] >= threshold) & (df["incorrect_predictions"] >= min_errors)
        outliers = df.loc[mask].copy()
        outliers["error_threshold"] = threshold  # helpful context in result

        if save_res:
            folder_path = self.pm.get_complex_ml_path(
                self.parameters, ["metadata_analysis", self.model_name], False, True, self.run_name
            )

            self.saver.save_file("dataframe", folder_path, "Summary", "outliers_by_groups", outliers)

        # Add to global outliers
        self.saver.save_global_outliers(self.parameters, outliers, "patient_id")

        return outliers

    def select_outlier_epochs(self, label: int = 1, save_res: bool = False) -> pd.DataFrame:
        """
        Identifies and retrieves epochs from a dataset that have been misclassified based
        on the given label. Optionally saves the results if specified.

        :param label: The target label to filter misclassified epochs, default is 1.
        :type label: int
        :param save_res: Indicates whether to save the results to an external location,
            default is False.
        :type save_res: bool
        :return: A dataframe containing the misclassified epochs.
        :rtype: pd.DataFrame
        """
        results_df = self.loader.load_results(self.parameters, self.run_name, self.model_name)

        # Get wrongly classified epochs with given label
        misclassified_df = results_df[
            (results_df["label"] == label) & (results_df["prediction"] != results_df["label"])
            ][["Start", "End", "ResultID"]]
        misclassified_df = pd.DataFrame(misclassified_df)  # Explicit cast, because IDE thinks it's a Series
        misclassified_df = misclassified_df.sort_values(by=["ResultID", "Start"]).reset_index(drop=True)

        if save_res:
            folder_path = self.pm.get_complex_ml_path(
                self.parameters, ["metadata_analysis", self.model_name], False, True, self.run_name
            )

            self.saver.save_file("dataframe", folder_path, "Summary", f"outlier_epochs_for_label_{label}", misclassified_df)

        # Add to global outliers
        self.saver.save_global_outliers(self.parameters, misclassified_df, "epoch")

        return misclassified_df

    @staticmethod
    def _compute_iqr_threshold(df: pd.DataFrame, iqr_multiplier: float):
        """
        Computes the IQR (Interquartile Range) threshold for a given DataFrame. The threshold
        is calculated using the "error_rate" column, where the upper boundary is adjusted
        based on the specified IQR multiplier. The computed threshold ensures a maximum
        value of 1.0 (100%).

        :param df: The DataFrame containing the "error_rate" column for calculation.
                   It must include the specified column for quantile computation.
        :type df: pd.DataFrame
        :param iqr_multiplier: A multiplier for the IQR to calculate the threshold.
                               Higher values result in a less restrictive threshold.
        :type iqr_multiplier: float
        :return: The computed IQR threshold with a maximum boundary of 1.0.
        :rtype: float
        """

        q1 = df["error_rate"].quantile(0.25)
        q3 = df["error_rate"].quantile(0.75)
        iqr = q3 - q1
        threshold = min(q3 + iqr_multiplier * iqr, 1.0)  # Maximum can't be higher than 100%
        return threshold