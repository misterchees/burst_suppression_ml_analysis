from MachineLearning.Evaluation.metrics_evaluator import MetricsEvaluator
from MachineLearning.IO.load_data import LoadData
from MachineLearning.Utils.config_handler import load_config
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def print_metrics(_run_name=None):
    loader = LoadData()
    evaluator = MetricsEvaluator(None, None, None, None, None)
    curent_params = load_config("parameters_config.yaml")["current_params"]
    print(f"Testing Parameters: {curent_params}\n")
    current_metrics = loader.load_metrics(curent_params, "svm", _run_name)
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



if __name__ == "__main__":
    run_name = "norm2_in_place_2"

    all_metrics = print_metrics(run_name)
    # individual_results = all_metrics["individual_results"]
    # plot_cell_distributions_percent(individual_results)
