import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class MetadataAnalyzer:
    """
    Analyzes the classification results in relation to metadata.
    Useful to explore structural performance differences beyond standard metrics.
    """

    def __init__(self, results_df: pd.DataFrame):
        """
        Initializes the MetadataAnalyzer with a DataFrame of results.
        At least a "label" and a "prediction" column must be present in the DataFrame.

        :param results_df: A DataFrame that includes columns for true and predicted labels,
                           and relevant metadata like 'Start', 'End', 'ResultID', etc.
        """
        self.df = results_df.copy()
        if "error" not in self.df.columns and "label" in self.df.columns and "prediction" in self.df.columns:
            self.df["error"] = (self.df["label"] != self.df["prediction"]).astype(int)

    def error_by_group(self, group_col: str) -> pd.Series:
        """
        Computes the mean classification error per group.

        :param group_col: Column name to group by (e.g., 'ResultID').
        :returns: A Series of average errors per group.
        """
        return self.df.groupby(group_col)["error"].mean()

    def class_distribution_by_group(self, group_col: str) -> pd.DataFrame:
        """
        Computes the true label distribution per group.

        :param group_col: Column name to group by.
        :returns: Crosstab of class proportions per group.
        """
        return pd.crosstab(self.df[group_col], self.df["label"], normalize="index")

    def correlation_with_error(self, method: str = "pearson") -> pd.Series:
        """
        Correlates numeric columns (e.g., metadata or features) with the classification error.

        :param method: Correlation method ('pearson', 'spearman', etc.).
                    It defaults to pearson correlation, which is for assumed linear relationships.
        :returns: A Series with correlations to the error column.
        """
        numeric_df = self.df.select_dtypes(include="number")
        return numeric_df.corr(method=method)["error"].sort_values(ascending=False)

    def plot_error_distribution(self, group_col: str, show_plt=True):
        """
        Creates a boxplot of error rates by group and shows the plot.

        :param group_col: Grouping column, e.g., 'ResultID'.
        :param show_plt: Boolean to enable/disable showing the plot.
        :returns: A matplotlib.pyplot.Figure object.
        """
        error_by_group = self.error_by_group(group_col).reset_index()
        error_by_group.columns = [group_col, "mean_error"]
        sns.boxplot(data=error_by_group, x=group_col, y="mean_error")
        plt.xticks(rotation=45)
        plt.title(f"Classification Error by {group_col}")
        plt.tight_layout()

        if show_plt:
            plt.show()

        return plt

    def plot_temporal_error(self, id_col: str, show_plt=True):
        """
        Plots error over time (Start) for each ResultID individually and shows the plot.

        :param id_col: Identifier column for separate series (e.g., 'ResultID').
        :param show_plt: Boolean to enable/disable showing the plot.
        :returns: A matplotlib.pyplot.Figure object.
        """
        for rid, group in self.df.groupby(id_col):
            plt.plot(group["Start"], group["error"], label=str(rid))
        plt.xlabel("Start Time")
        plt.ylabel("Classification Error (0/1)")
        plt.title("Temporal Error Progression per Recording")
        plt.legend()
        plt.tight_layout()

        if show_plt:
            plt.show()

        return plt

    def confusion_matrix_by_group(self, group_col: str) -> dict:
        """
        Generates a confusion matrix for each group separately.

        :param group_col: Column to group by (e.g., 'ResultID').
        :returns: Dictionary of confusion matrices per group.
        """
        from sklearn.metrics import confusion_matrix
        grouped_matrices = {}
        for group, group_df in self.df.groupby(group_col):
            cm = confusion_matrix(group_df["label"], group_df["prediction"], labels=sorted(self.df["label"].unique()))
            grouped_matrices[group] = cm
        return grouped_matrices
