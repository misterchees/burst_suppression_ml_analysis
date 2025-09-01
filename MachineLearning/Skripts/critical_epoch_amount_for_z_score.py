import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from MachineLearning.Models.svm_classifier import SVMClassifier
import matplotlib
matplotlib.use('TkAgg')

def run_sampling_experiment(train_df, test_df, label_col, feature_cols, sample_sizes, n_iterations=10,
                            metrics=("accuracy", "f1", "precision", "recall"), sample_set="train", use_implemented_svm=True,
                            svm_c = 1.0, svm_kernel = "rbf"):
    """
    Run repeated sampling experiments with columnwise Z-Score normalization. The train set size stays always the same.
    The sampling only provides the basis for calculation of the mean and variance of the z-score normalization
    of the whole sample.

    :param train_df: Training DataFrame
    :param test_df: Test DataFrame
    :param label_col: Name of label column
    :param feature_cols: List of feature columns to use
    :param sample_sizes: List of sample sizes (total per iteration, evenly split between classes)
    :param n_iterations: Number of iterations per sample size
    :param metrics: Tuple of metrics to calculate ("accuracy", "f1", "precision", "recall")
    :param sample_set: Indicates from which set to sample for normalization parameters:
        "all" for all episodes, "train" for training set, or "test" for test set.
    :param use_implemented_svm: If True, use the implemented SVM classifier. If False, use sklearn's SVC.
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

            # Normalize train & test with calculated mean and std from sampled df
            train_norm = _normalize_features_in_df(train_df, sample_df, _normalize_array, feature_cols)
            test_norm = _normalize_features_in_df(test_df, sample_df, _normalize_array, feature_cols)

            if use_implemented_svm:
                clf = SVMClassifier(C=svm_c, kernel=svm_kernel)
                clf.train(train_norm, train_df[label_col])
                preds = clf.predict(test_norm)
            else:
                # Train SVM
                clf = SVC(kernel=svm_kernel, C=svm_c)
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

def run_sampling_experiment2(train_df, test_df, label_col, feature_cols, sample_sizes, n_iterations=10,
                            metrics=("accuracy", "f1", "precision", "recall"),svm_c = 1.0, svm_kernel = "rbf"):
    """
    Sample train subset from train set to train classifier on it and classify always with the same test set.
    :param train_df:
    :param test_df:
    :param label_col:
    :param feature_cols:
    :param sample_sizes:
    :param n_iterations:
    :param metrics:
    :param svm_c:
    :param svm_kernel:
    :return:
    """
    results = {m: {s: [] for s in sample_sizes} for m in metrics}
    sampling_df = train_df.copy()

    for s in sample_sizes:
        half_s = s // 2
        for i in range(n_iterations):
            # Stratified sampling: half from label 0, half from label 1
            class0 = sampling_df[sampling_df[label_col] == 0].sample(half_s, replace=False, random_state=i)
            class1 = sampling_df[sampling_df[label_col] == 1].sample(half_s, replace=False, random_state=i)
            # Create new train set from sampled df
            train_df_sample = pd.concat([class0, class1])

            # Normalize train & test with calculated mean and std from sampled df
            train_norm = _normalize_features_in_df(train_df_sample, train_df_sample, _normalize_array, feature_cols)
            test_norm = _normalize_features_in_df(test_df, train_df_sample, _normalize_array, feature_cols)

            # Train SVM
            clf = SVC(kernel=svm_kernel, C=svm_c)
            clf.fit(train_norm, train_df_sample[label_col])
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

def run_classification_without_sampling(use_implemented_svm=False, label_col="label", metrics=("accuracy", "f1", "precision", "recall")):
    train_df = pd.read_csv("D:\\Daten\\Test_and_train\\Splits\\Splits_70_080_20_5\\Summary_Episodes_20_000\\norm_feat_over_vector_0\\4_6_train_split.csv").copy()
    train_labels = train_df[label_col].values
    train_df = train_df.drop(columns=["Start", "End", "ResultID", label_col])
    test_df = pd.read_csv("D:\\Daten\\Test_and_train\\Splits\\Splits_70_080_20_5\\Summary_Episodes_20_000\\norm_feat_over_vector_0\\4_6_test_split.csv").copy()
    test_labels = test_df[label_col].values
    test_df = test_df.drop(columns=["Start", "End", "ResultID", label_col])


    if use_implemented_svm:
        clf = SVMClassifier(C=1, kernel="rbf")
        clf.train(train_df, train_labels)
        preds = clf.predict(test_df)
    else:
        # Train SVM
        clf = SVC(kernel="rbf", C=1)
        clf.fit(train_df, train_labels)
        preds = clf.predict(test_df)

    # Compute metrics
    if "accuracy" in metrics:
        print(f"Acc: {accuracy_score(test_labels, preds)}\n")
    if "f1" in metrics:
        print(f"f1: {f1_score(test_labels, preds)}\n")
    if "precision" in metrics:
        print(f"precision: {precision_score(test_labels, preds)}\n")
    if "recall" in metrics:
        print(f"recall: {recall_score(test_labels, preds)}\n")


def _normalize_array(values: np.ndarray, sample_for_vars: np.ndarray) -> np.ndarray:
    """Normalizes an array of values based on the method given in the class and an optional feature type."""
    to_normalize = np.asarray(values)

    return (to_normalize - np.mean(sample_for_vars)) / np.std(sample_for_vars)


def _normalize_features_in_df(df, sample_df, normalizing_function, feature_cols):
    """
    Normalize all feature columns in a DataFrame column-wise using a given normalization function.
    Keeps Start, End, ResultID unchanged.

    :param df: pandas.DataFrame with columns [feature1, feature2, ...]
    :param normalizing_function: function that takes a numpy array and returns a normalized numpy array
    :return: pandas.DataFrame with normalized features
    """

    df_norm = df.copy()
    for col in feature_cols:
        df_norm[col] = normalizing_function(df[col].values, sample_df[col].values)

    return df_norm

if __name__ == "__main__":
    split_folder = "D:\\Daten\\Other\\Splits_for_normalization_statistics\\"
    train_df_ = pd.read_parquet(f"{split_folder}train_set.parquet").copy()
    train_df_ = train_df_.drop(columns=["Start", "End", "ResultID"])
    test_df_ = (pd.read_parquet(f"{split_folder}test_set.parquet")).copy()
    test_df_ = test_df_.drop(columns=["Start", "End", "ResultID"])

    feature_cols_ = ["Delta", "Theta", "Alpha", "Beta", "Spectral_skewness", "Spectral_kurtosis", "Shannon_entropy", "Permutation_entropy"]
    metrics_to_plot_ = ["accuracy", "f1", "precision", "recall"]

    # results_ = run_sampling_experiment(train_df_, test_df_,"label", feature_cols_, sample_sizes=[2, 3, 4 ,5, 8, 10, 12, 15, 20],
    #                                    n_iterations=50, sample_set="train", use_implemented_svm=False, svm_c=1, svm_kernel="rbf")
    results_ = run_sampling_experiment2(train_df_, test_df_, "label", feature_cols_,
                                       sample_sizes=[10, 50, 100, 150, 200, 250, 300, 350],
                                       n_iterations=50, svm_c=1,
                                       svm_kernel="rbf")
    plot_results(results_, metrics_to_plot_)
    # run_classification_without_sampling()