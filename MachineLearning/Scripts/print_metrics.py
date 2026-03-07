from MachineLearning.Evaluation.metrics_evaluator import MetricsEvaluator
from MachineLearning.IO.load_data import LoadData
from MachineLearning.Utils.config_handler import load_config
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from MachineLearning.Utils.path_manager import PathManager

pm = PathManager()

def print_metrics(_run_name=None):
    loader = LoadData(pm)
    evaluator = MetricsEvaluator(None, None, None, None, None)
    curent_params = load_config("parameters_config.yaml")["current_params"]
    print(f"Testing Parameters: {curent_params}\n")

    # Load metrics
    folder_path = pm.get_complex_ml_path(
        curent_params, ["results", "svm"], False, True, _run_name)
    current_metrics = loader.load_json(folder_path / "folds_metrics.json")
    evaluator.print_result(current_metrics["summary"], True)
    return current_metrics


def plot_confusion_matrices(results):
    n_folds = len(results)
    fig, axes = plt.subplots(2, (n_folds+1)//2, figsize=(15, 8))
    axes = axes.ravel()

    for i, res in enumerate(results):
        cm = np.array(res['confusion_matrix'][1:])[:,1:].astype(int)  # Extrahiert die 2x2-Matrix
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["faw (0)", "awake (1)"],
                    yticklabels=["faw (0)", "awake (1)"],
                    ax=axes[i])
        axes[i].set_title(f"{res['result']} (Acc={res['accuracy']:.2f})")

    plt.tight_layout()
    plt.show()


def plot_aggregate_confusion_matrix(results):
    cms = []
    for res in results:
        cm = np.array(res['confusion_matrix'][1:])[:,1:].astype(int)
        cms.append(cm)
    cms = np.array(cms)

    mean_cm = cms.mean(axis=0)
    std_cm = cms.std(axis=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(mean_cm, annot=True, fmt=".1f", cmap="Blues", cbar=False,
                xticklabels=["faw (0)", "awake (1)"],
                yticklabels=["faw (0)", "awake (1)"], ax=ax)

    ax.set_title("Mean Confusion Matrix across folds")
    for i in range(mean_cm.shape[0]):
        for j in range(mean_cm.shape[1]):
            ax.text(j+0.5, i+0.7, f"±{std_cm[i,j]:.1f}",
                    ha="center", va="center", color="red", fontsize=8)

    plt.show()


def plot_cell_distributions_percent(results):
    """
    Plots boxplots of confusion matrix cell distributions (in %) across folds.
    Normalization is done row-wise so each row sums to 100%.
    """
    data = []
    for res in results:
        cm = res['confusion_matrix']

        # Falls Confusion Matrix als DataFrame vorliegt → nur Werte
        if hasattr(cm, "values"):
            cm = cm.values
        cm = np.array(cm)

        # Falls Labels drin sind (mehr als 2x2) → nur unteren 2x2 Block nehmen
        if cm.shape[0] > 2 or cm.shape[1] > 2:
            cm = cm[-2:, -2:]

        cm = cm.astype(float)

        # Zeilenweise Normalisierung auf 100%
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_percent = (cm / row_sums) * 100

        data.append({
            "fold": res['result'],
            "True faw, Pred faw": cm_percent[0,0],
            "True faw, Pred awake": cm_percent[0,1],
            "True awake, Pred faw": cm_percent[1,0],
            "True awake, Pred awake": cm_percent[1,1],
        })

    df = pd.DataFrame(data).melt(id_vars="fold", var_name="cell", value_name="percent")

    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x="cell", y="percent")
    sns.stripplot(data=df, x="cell", y="percent", hue="fold", dodge=True, alpha=0.5)
    plt.xticks(rotation=30)
    plt.ylabel("Percent (%)")
    plt.title("Confusion Matrix Cell Distributions Across Folds (Row-normalized)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

def plot_run_metrics(list_of_runs, run_names_=None, metrics_to_plot=None,
                     plot_confusion=False, plot_type="bar"):
    """
    Plots summary metrics across multiple runs.

    :param list_of_runs: List of summary_dicts (each run).
    :param run_names_: List of names/labels for runs. If None, indices are used.
    :param metrics_to_plot: List of metrics to plot (subset of ["accuracy","precision","recall","f1"]).
    :param plot_confusion: If True, also plots confusion matrices for each run.
    :param plot_type: "bar", "box", or "violin".
    """
    if run_names_ is None:
        run_names_ = [f"Run_{i + 1}" for i in range(len(list_of_runs))]

    if metrics_to_plot is None:
        metrics_to_plot = ["accuracy", "precision", "recall", "f1"]

    # --- Collect scalar metrics (acc, pre, rec, f1) ---
    metric_data = []
    for run_name_, run_dict in zip(run_names_, list_of_runs):
        for metric in metrics_to_plot:
            if metric in run_dict:
                mean_val = run_dict[metric]["mean"] * 100
                var_val = run_dict[metric]["standard_deviation"] * 100
                run_name_ = run_name_.removeprefix("zscorenorm2_")
                metric_data.append({
                    "Run": run_name_,
                    "Metric": metric.capitalize(),
                    "Mean": mean_val,
                    "Variance": var_val
                })

    df_metrics = pd.DataFrame(metric_data)

    if df_metrics.empty:
        print("No matching metrics found in list_of_runs.")
        return

    plt.figure(figsize=(10, 6))

    if plot_type == "bar":
        ax = sns.barplot(data=df_metrics, x="Run", y="Mean", hue="Metric", errorbar="sd")
        plt.xticks(rotation=90)
        plt.ylabel("Score (%)")
        plt.title("Run Metrics (Mean ± SD)")
        plt.legend(title="Metric")
        # x axis: Rotate labels, replace _ with space and abbreviate features further
        labels = [_prettify_feature_name(t.get_text()) for t in ax.get_xticklabels()]
        ax.set_xticklabels(labels, rotation=45, ha="right")

        # --- Separator line in the middle (only for bar plot) ---
        n_runs = len(run_names_)
        separator_pos = n_runs / 2 - 0.5
        plt.axvline(x=separator_pos, color="black", linestyle="--")

        # Values over bars
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.text(
                    p.get_x() + p.get_width() / 2,  # x middle of the bar
                    height,                         # y bar height
                    f"{height:.1f}",                # One value after delimiter
                    ha="center", va="bottom",       # center over bar
                    fontsize=8
                )


    elif plot_type == "box":
        sns.boxplot(data=df_metrics, x="Metric", y="Mean")
        sns.stripplot(data=df_metrics, x="Metric", y="Mean", color="black", alpha=0.3)
        plt.ylabel("Score (%)")
        plt.title("Metric Distributions Across Runs")

    elif plot_type == "violin":
        sns.violinplot(data=df_metrics, x="Metric", y="Mean", inner="quartile")
        sns.stripplot(data=df_metrics, x="Metric", y="Mean", color="black", alpha=0.3)
        plt.ylabel("Score (%)")
        plt.title("Metric Distributions Across Runs")

    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")

    plt.tight_layout()
    plt.show()

    # --- Plot confusion matrices (optional) ---
    if plot_confusion:
        for run_name_, run_dict in zip(run_names_, list_of_runs):
            cm_mean = run_dict["confusion_matrix"]["mean"].astype(float)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm_mean, annot=True, fmt=".1f", cmap="Blues", cbar=False)
            plt.title(f"Confusion Matrix (mean %) - {run_name_}")
            plt.ylabel("True")
            plt.xlabel("Predicted")
            plt.tight_layout()
            plt.show()


def _prettify_feature_name(name: str) -> str:
    """
    Maps raw feature codes to pretty labels for plotting.

    :param name: Feature string (e.g., "bandAT_skew").
    :return: Pretty formatted label.
    """
    mapping = {
        "kurt": "KU",
        "skew": "SK",
        "perm": "PE",
        "shan": "SE",
    }

    parts = name.split("_")
    pretty_parts = []

    for part in parts:
        if part.startswith("band") and len(part) > 4:
            # Extract band letters (e.g. "bandAT" -> "AT")
            bands = part[4:]
            # Convert to "Band(A, T)" etc.
            pretty_parts.append(f"Band({', '.join(bands)})")
        elif part in mapping:
            pretty_parts.append(mapping[part])
        else:
            # Fallback: leave as is
            pretty_parts.append(part)

    return ", ".join(pretty_parts)


def select_top_bottom_runs(list_of_runs, run_names_=None, metric="accuracy", top_n=5):
    """
    Selects top and bottom runs based on a given metric.

    :param list_of_runs: List of summary_dicts (each run).
    :param run_names_: Optional list of names.
    :param metric: Metric to rank runs by.
    :param top_n: Number of top and bottom runs to return.
    :returns: (selected_runs, selected_names)
    """
    if run_names_ is None:
        run_names_ = [f"Run_{i + 1}" for i in range(len(list_of_runs))]

    scores = []
    for run_name_, run_dict in zip(run_names_, list_of_runs):
        if metric in run_dict:
            scores.append((run_name_, run_dict, run_dict[metric]["mean"]))

    df = pd.DataFrame(scores, columns=["Run", "Dict", "Score"])
    df = df.sort_values("Score", ascending=False)

    top = df.head(top_n)
    bottom = df.tail(top_n)

    selected = pd.concat([top, bottom])
    return list(selected["Dict"]), list(selected["Run"])


def collect_run_metrics(list_of_runs, run_names_=None, metrics_to_collect=None):
    """
    Collects summary metrics across multiple runs into a DataFrame.

    :param list_of_runs: List of summary_dicts (each run).
    :param run_names_: List of names/labels for runs. If None, indices are used.
    :param metrics_to_collect: List of metrics to collect (subset of ["accuracy","precision","recall","f1"]).
    :return: pd.DataFrame with one row per run and metrics as columns.
    """
    if run_names_ is None:
        run_names_ = [f"Run_{i + 1}" for i in range(len(list_of_runs))]

    if metrics_to_collect is None:
        metrics_to_collect = ["accuracy", "precision", "recall", "f1"]

    rows = []
    for run_name_, run_dict in zip(run_names_, list_of_runs):
        clean_name = run_name_.removeprefix("zscorenorm2_")
        row = {"Run": clean_name}
        for metric in metrics_to_collect:
            if metric in run_dict:
                row[metric.capitalize()] = run_dict[metric]["mean"] * 100
        rows.append(row)

    df = pd.DataFrame(rows)
    return df

def best_and_worst_per_feature_count(df, metric="Accuracy"):
    """
    For each number of features, return the best and worst runs based on a selected metric.

    :param df: DataFrame with at least columns ["Run", metric].
    :param metric: Metric name to evaluate (e.g., "Accuracy", "Precision", "Recall", "F1").
    :return: DataFrame summarizing best and worst runs per feature count.
    """

    def count_features(run_name_):
        """Count features in a run name (special case: bandX... expands to multiple)."""
        parts = run_name_.split("_")
        total = 0
        for p in parts:
            if p.startswith("band"):
                total += len(p.replace("band", ""))  # each letter counts as one feature
            else:
                total += 1
        return total

    df = df.copy()
    df["NumFeatures"] = df["Run"].apply(count_features)

    results = []
    for num_feats, group in df.groupby("NumFeatures"):
        # best
        best_row = group.loc[group[metric].idxmax()]
        # worst
        worst_row = group.loc[group[metric].idxmin()]

        results.append({
            "NumFeatures": num_feats,
            "BestRun": best_row["Run"],
            "BestScore": best_row[metric],
            "WorstRun": worst_row["Run"],
            "WorstScore": worst_row[metric],
        })

    return pd.DataFrame(results).sort_values("NumFeatures").reset_index(drop=True)



if __name__ == "__main__":
    run_name = "norm2_z_score_0"
    # from MachineLearning.Core.runner import generate_feature_combinations
    # feat_comb = generate_feature_combinations()
    # run_names = [rn for _,_,rn in feat_comb]
    # summary_metrics = []
    # for run_name in run_names:
    #     summary_metrics.append(print_metrics(run_name)["summary"])

    # # Create dataframes with subset runs
    # collected_run_metrics = collect_run_metrics(summary_metrics, run_names, metrics_to_collect=["accuracy", "precision", "recall", "f1"])
    # PathUtils.save_file_as_csv(collected_run_metrics, r"D:\Daten\Further_analysis\Subset_runs\subset_run_metrics.csv", False)
    # metric = "Accuracy"
    # best_worst_acc_df = best_and_worst_per_feature_count(collected_run_metrics, metric=metric)
    # PathUtils.save_file_as_csv(best_worst_acc_df, r"D:\Daten\Further_analysis\Subset_runs\best_worst_acc_df.csv", False)
    # metric = "F1"
    # best_worst_f1_df = best_and_worst_per_feature_count(collected_run_metrics, metric=metric)
    # PathUtils.save_file_as_csv(best_worst_f1_df, r"D:\Daten\Further_analysis\Subset_runs\best_worst_f1_df.csv", False)

    # Create plots with run data
    # plot_run_metrics(summary_metrics, run_names, plot_type="violin")
    # metric = "precision"
    # selected_dicts, selected_names = select_top_bottom_runs(summary_metrics, run_names, metric=metric, top_n=3)
    # plot_run_metrics(selected_dicts, selected_names,metrics_to_plot=[metric], plot_type="bar")
    # metric = "accuracy"
    # selected_dicts, selected_names = select_top_bottom_runs(summary_metrics, run_names, metric=metric, top_n=3)
    # plot_run_metrics(selected_dicts, selected_names, metrics_to_plot=[metric], plot_type="bar")
    # metric = "recall"
    # selected_dicts, selected_names = select_top_bottom_runs(summary_metrics, run_names, metric=metric, top_n=3)
    # plot_run_metrics(selected_dicts, selected_names,metrics_to_plot=[metric], plot_type="bar")
    # metric = "f1"
    # selected_dicts, selected_names = select_top_bottom_runs(summary_metrics, run_names, metric=metric, top_n=3)
    # plot_run_metrics(selected_dicts, selected_names,metrics_to_plot=[metric], plot_type="bar")

    all_metrics = print_metrics(run_name)
    # individual_results = all_metrics["individual_results"]
    # plot_cell_distributions_percent(individual_results)
