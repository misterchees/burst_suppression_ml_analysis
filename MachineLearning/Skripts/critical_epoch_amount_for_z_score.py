import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

def run_sampling_experiment(train_df, test_df, label_col, feature_cols, sample_sizes, n_iterations=10,
                            metrics=("accuracy", "f1", "precision", "recall"), sample_set="train"):
    """
    Run repeated sampling experiments with columnwise Z-Score normalization.

    :param train_df: Training DataFrame
    :param test_df: Test DataFrame
    :param label_col: Name of label column
    :param feature_cols: List of feature columns to use
    :param sample_sizes: List of sample sizes (total per iteration, evenly split between classes)
    :param n_iterations: Number of iterations per sample size
    :param metrics: Tuple of metrics to calculate ("accuracy", "f1", "precision", "recall")
    :param sample_set: Indicates from which set to sample for normalization parameters:
        "all" for all episodes, "train" for training set, or "test" for test set.
    :return: dict of results per sample size and metric
    """
    results = {m: {s: [] for s in sample_sizes} for m in metrics}

    if sample_set == "all":
        class_0_df = pd.read_csv("D:\\Daten\\Other\\Splits_for_normalization_statistics\\Summary_Episodes_20_000.csv").copy()
        class_1_df = pd.read_csv("D:\\Daten\\Other\\Splits_for_normalization_statistics\\Awake_20.csv").copy()
        class_0_df["label"] = 0
        class_1_df["label"] = 1
        sampling_df = pd.concat([class_0_df, class_1_df], ignore_index=True)
    elif sample_set == "train":
        sampling_df = train_df.copy()
    elif sample_set == "test":
        sampling_df = test_df.copy()
    else:
        raise ValueError("sample_set must be 'all', 'train', or 'test'")

    for s in sample_sizes:
        half_s = s // 2
        for i in range(n_iterations):
            # Stratified sampling: half from label 0, half from label 1
            class0 = sampling_df[sampling_df[label_col] == 0].sample(half_s, replace=False, random_state=i)
            class1 = sampling_df[sampling_df[label_col] == 1].sample(half_s, replace=False, random_state=i)
            sample_df = pd.concat([class0, class1])

            # Compute mean & std per column
            means = sample_df[feature_cols].mean()
            stds = sample_df[feature_cols].std(ddof=0)

            # Normalize train & test
            train_norm = (train_df[feature_cols] - means) / stds
            test_norm = (test_df[feature_cols] - means) / stds

            # Train SVM
            clf = SVC(kernel="rbf")
            clf.fit(train_norm, train_df[label_col])
            preds = clf.predict(test_norm)

            # Compute metrics
            if "accuracy" in metrics:
                results["accuracy"][s].append(accuracy_score(test_df[label_col], preds))
                print(f"Acc: {accuracy_score(test_df[label_col], preds)} for sample size {s} and iteration {i} \n")
            if "f1" in metrics:
                results["f1"][s].append(f1_score(test_df[label_col], preds))
            if "precision" in metrics:
                results["precision"][s].append(precision_score(test_df[label_col], preds))
            if "recall" in metrics:
                results["recall"][s].append(recall_score(test_df[label_col], preds))

    return results


def plot_results(results, metrics_to_plot=None):
    """
    Plot mean ± std error for selected metrics.

    :param results: dict returned by run_sampling_experiment
    :param metrics_to_plot: list of metrics to plot (subset of keys in results). If None, plot all.
    """
    if metrics_to_plot is None:
        metrics_to_plot = list(results.keys())

    for m in metrics_to_plot:
        means = [np.mean(results[m][s]) for s in results[m]]
        stderrs = [np.std(results[m][s]) / np.sqrt(len(results[m][s])) for s in results[m]]

        plt.errorbar(list(results[m].keys()), means, yerr=stderrs, capsize=5, label=m)

    plt.xlabel("Sample size (per subset)")
    plt.ylabel("Score")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    split_folder = "D:\\Daten\\Other\\Splits_for_normalization_statistics\\"
    train_df_ = pd.read_parquet(f"{split_folder}train_set.parquet").copy()
    train_df_ = train_df_.drop(columns=["Start", "End", "ResultID"])
    test_df_ = (pd.read_parquet(f"{split_folder}test_set.parquet")).copy()
    test_df_ = test_df_.drop(columns=["Start", "End", "ResultID"])

    feature_cols_ = ["Delta", "Theta", "Alpha", "Beta", "Spectral_skewness", "Spectral_kurtosis", "Shannon_entropy", "Permutation_entropy"]
    metrics_to_plot_ = ["accuracy", "f1", "precision", "recall"]

    results_ = run_sampling_experiment(train_df_, test_df_,"label",
                                       feature_cols_, sample_sizes=[100, 1000, 1500], n_iterations=100, sample_set="all")
    plot_results(results_, metrics_to_plot_)