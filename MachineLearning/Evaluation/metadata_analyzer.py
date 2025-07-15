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

    def error_by_group(self, group_col: str) -> pd.DataFrame:
        """
        Computes the mean classification error per group,
        and returns total and incorrect counts as well.

        :param group_col: Column name to group by (e.g., 'ResultID').
        :returns: DataFrame with total count, error count, and error rate per group.
        """
        grouped = self.df.groupby(group_col)
        total = grouped.size()
        errors = grouped["error"].sum()  # Sums all errors per group (This works since every error is represented as 1)
        error_rate = errors / total

        return pd.DataFrame({
            "total_samples": total,
            "incorrect_predictions": errors,
            "error_rate": error_rate
        })

    def error_for_label_by_group(self, group_col: str, target_label: int | str = 1) -> pd.DataFrame:
        """
        Computes total samples, wrong predictions, and error rate for a single
        label within each group (e.g. ResultID).

        :param group_col: Column name to group by (e.g. 'ResultID').
        :param target_label: The label value whose errors should be analyzed.
        :returns: DataFrame with total count, incorrect count, and error rate –
                  restricted to rows where self.df['label'] == target_label.
        """
        subset = self.df[self.df["label"] == target_label]  # Subset of target label

        grouped = subset.groupby(group_col)
        total = grouped.size()
        errors = grouped["error"].sum()  # Sum up all errors per group
        error_rate = errors / total

        return pd.DataFrame({
            "total_samples_label": total,
            "incorrect_predictions_label": errors,
            "error_rate_label": error_rate
        })

    def class_distribution_by_group(self, group_col: str) -> pd.DataFrame:
        """
        Computes both class counts and relative class distribution per group.

        :param group_col: Column name to group by.
        :returns: DataFrame with absolute and relative class distributions.
        """
        abs_counts = pd.crosstab(self.df[group_col], self.df["label"])
        rel_props = pd.crosstab(self.df[group_col], self.df["label"], normalize="index")

        # Kombiniere beide mit mehrstufiger Spaltenüberschrift
        abs_counts.columns = pd.MultiIndex.from_product([["abs"], abs_counts.columns])
        rel_props.columns = pd.MultiIndex.from_product([["rel"], rel_props.columns])

        return pd.concat([abs_counts, rel_props], axis=1)

    def correlation_with_error(self, method: str = "pearson") -> pd.Series:
        """
        Correlates numeric columns (e.g., metadata or features) with the classification error.

        :param method: Correlation method ('pearson', 'spearman', etc.).
                    It defaults to pearson correlation, which is for assumed linear relationships.
        :returns: A Series with correlations to the error column.
        """
        numeric_df = self.df.select_dtypes(include="number")
        return numeric_df.corr(method=method)["error"].sort_values(ascending=False)

    def plot_error_distribution(self, group_col: str, show_plt: bool = True):
        """
        Creates a boxplot of error rates by group, excluding perfect groups (error=0).

        :param group_col: Grouping column, e.g., 'ResultID'.
        :param show_plt: Whether to show the plot.
        :returns: A matplotlib Figure object.
        """
        df = self.error_by_group(group_col).reset_index()
        df = df[df["error_rate"] > 0]  # exclude perfect groups

        if df.empty:
            print("No groups with errors found. Nothing to plot.")
            return None

        # TO DO: Change to bar plot (the vertical bars)
        fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.4), 5))
        sns.boxplot(data=df, x=group_col, y="error_rate", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_title(f"Classification Error by {group_col}")
        fig.tight_layout()

        if show_plt:
            plt.show()

        return fig

    def plot_temporal_error(self, id_col: str, show_plt: bool = True, max_series: int = 10):
        """
        Plots error over time for the top-N groups with the highest average error.

        :param id_col: Identifier column, e.g., 'ResultID'.
        :param show_plt: Whether to show the plot.
        :param max_series: Max number of series to plot to avoid overcrowding.
        :returns: A matplotlib Figure object.
        """
        avg_errors = self.error_by_group(id_col).sort_values("error_rate", ascending=False)
        top_ids = avg_errors.head(max_series).index

        fig, ax = plt.subplots(figsize=(10, 5))
        for rid in top_ids:
            subset = self.df[self.df[id_col] == rid]
            ax.plot(subset["Start"], subset["error"], label=str(rid), alpha=0.7)

        ax.set_xlabel("Start Time")
        ax.set_ylabel("Classification Error (0/1)")
        ax.set_title(f"Temporal Error Progression – Top {max_series} {id_col}")
        ax.legend(title=id_col, bbox_to_anchor=(1.05, 1), loc="upper left")
        fig.tight_layout()

        if show_plt:
            plt.show()

        return fig

    def confusion_matrix_by_group(self, group_col: str, class_0_name: str = "faw", class_1_name: str = "awake") -> dict:
        """
        Generates a labeled confusion matrix for each group separately.

        :param group_col: Column to group by (e.g., 'ResultID').
        :param class_0_name: Human-readable name for class 0.
        :param class_1_name: Human-readable name for class 1.
        :returns: Dictionary of labeled confusion matrices per group as DataFrames.
        """
        from sklearn.metrics import confusion_matrix

        # Ensure deterministic label order
        class_labels = [0, 1]
        row_labels = [f"True {class_0_name} (0)", f"True {class_1_name} (1)"]
        col_labels = [f"Predicted {class_0_name} (0)", f"Predicted {class_1_name} (1)"]

        grouped_matrices = {}
        for group, group_df in self.df.groupby(group_col):
            cm = confusion_matrix(group_df["label"], group_df["prediction"], labels=class_labels)
            cm_df = pd.DataFrame(cm, index=row_labels, columns=col_labels)
            grouped_matrices[group] = cm_df

        return grouped_matrices

