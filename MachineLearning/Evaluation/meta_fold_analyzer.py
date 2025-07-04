"""This Module contains the MetaFoldAnalyzer class."""
import os
import pandas as pd
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns

from MachineLearning.IO.io_core import IOCore


class MetaFoldAnalyzer:
    """Class that calculates from results and metadata analysis from single folds overall statistics and trends."""
    def __init__(self, model_key: str, parameters: dict):
        """
        Initializes the MetaFoldAnalyzer instance with model and paths to calculated results and metadata analysis
        of single folds.

        :param model_key: The key of the model to analyze.
        :param parameters: A dictionary containing the parameters of the epochs from which the results were
                           calculated.
        """
        self.model_name = model_key

        io_basics = IOCore()
        # Set paths
        self.ml_results_path = io_basics.return_all_parameter_fullpath(parameters, False, False, "results", model_key)
        self.metadata_path = io_basics.return_all_parameter_fullpath(
            parameters, False, False, "metadata_analysis", model_key)

        # Container for analysis data
        self.fold_errors_by_group = {}
        self.fold_class_distributions = {}
        self.fold_metrics = {}

    def load_all_folds(self, group_col: str):
        """
        Load all relevant data (error_by_group, class_dist, metrics) from directories.
        """
        # Search for all folds with labels and errors
        fold_files = glob.glob(os.path.join(self.ml_results_path, "*full_and_pred.csv"))
        for fold_file in fold_files:
            fold_lname = os.path.basename(fold_file).replace(".csv", "")  # long name of fold
            fold_sname = fold_lname.replace("full_and_pred", "")  # short name of fold

            # Load all files containing error by group analysis for folds
            err_path = os.path.join(self.metadata_path, f"{fold_lname}_error_by_{group_col}.csv")
            if os.path.exists(err_path):
                self.fold_errors_by_group[fold_sname] = pd.read_csv(err_path, index_col=0)

            # Load all files containing class distribution by group
            dist_path = os.path.join(self.metadata_path, f"{fold_lname}_class_dist_per_{group_col}.csv")
            if os.path.exists(dist_path):
                self.fold_class_distributions[fold_sname] = pd.read_csv(dist_path, header=[0,1], index_col=0)

            # Load metrics for folds
            metrics_path = os.path.join(self.ml_results_path, "folds_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    self.fold_metrics[fold_sname] = json.load(f)

    def aggregate_error_by_group(self):
        """Returns a combined dataframe of errors by group (e.g. ResultID) over all folds."""
        combined = []
        for fold_name, df in self.fold_errors_by_group.items():
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
                # TO DO: path to metric is -> key: individual_results -> val: list with individual metric dicts
                # -> Search for metric with val result_name for key ["result"]
                # -> And now you can .get(metric, None) to get it.
                metric_val = self.fold_metrics[fold_name].get(metric, None)
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
            print("Keine Fehlerdaten vorhanden.")
            return None

        pivot = agg.pivot(index="fold", columns=group_col_name, values="error_rate")
        fig, ax = plt.subplots(figsize=(min(18, pivot.shape[1]*0.7), 6))
        sns.heatmap(pivot, annot=False, cmap="Reds", ax=ax)
        ax.set_title("Fehlerrate pro Fold und Gruppe")
        plt.tight_layout()

        if show_plt:
            plt.show()

        return fig
