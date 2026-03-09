import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu
from MachineLearning.IO.load_data import LoadData
from MachineLearning.IO.save_result import SaveResult
from MachineLearning.Utils.path_manager import PathManager

pm = PathManager()
saver = SaveResult(pm)
loader = LoadData(pm)

def analyze_center_of_mass_sets(hyperparameters: dict, confidence: float, class_a: int, class_b: int, test="t-test"):

    confidence_str = str(confidence).replace(".", "")
    a_file_name = f"PCA_clusterlabel_{class_a}_region_with_confidence_{confidence_str}_dims_2.csv"
    b_file_name = f"PCA_clusterlabel_{class_b}_region_with_confidence_{confidence_str}_dims_2.csv"
    a_df = loader.load_further_results(hyperparameters, "pca", a_file_name)
    b_df = loader.load_further_results(hyperparameters, "pca", b_file_name)
    diff_df = analyze_feature_differences(a_df, b_df, test=test)
    print(f"###############\n {diff_df}")
    class_dict = {0: "faw", 1: "correct_awake", 2: "wrong_awake"}
    a_name = class_dict[class_a]
    b_name = class_dict[class_b]

    # Save results
    folder_path = pm.get_complex_ml_path(
        hyperparameters, ["further_analysis", "stat_diff"], False, True
    )
    prefix = f"center_of_mass_sets_{a_name}_vs_{b_name}"
    suffix = f"statistical_differences_confidence_{confidence_str}"
    saver.save_file("dataframe", folder_path, prefix, suffix, diff_df)



def cohen_d(x, y):
    """
    Compute Cohen's d for independent samples.
    :param x: 1D array-like
    :param y: 1D array-like
    :return: float (effect size)
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else np.nan


def analyze_feature_differences(df1, df2, test="t-test"):
    """
    Analyze statistical differences between two DataFrames with the same features.
    Ignores metadata columns: Start, End, ResultID, label, and index if present.

    :param df1: pandas.DataFrame
    :param df2: pandas.DataFrame
    :param test: "t-test" or "mannwhitney"
    :return: pandas.DataFrame with feature statistics
    """
    ignore_cols = {"Start", "End", "ResultID", "label", "Unnamed: 0"}
    features = [col for col in df1.columns if col not in ignore_cols]

    results = []

    for feature in features:
        x = df1[feature].dropna().values
        y = df2[feature].dropna().values

        if test == "t-test":
            stat, p = ttest_ind(x, y, equal_var=False)  # Welch’s t-test (robust)
            d = cohen_d(x, y)
        elif test == "mannwhitney":
            stat, p = mannwhitneyu(x, y, alternative="two-sided")
            d = np.nan  # Cohen's d nicht definiert für Mann-Whitney
        else:
            raise ValueError("test must be 't-test' or 'mannwhitney'")

        results.append({
            "feature": feature,
            "mean_df1": np.mean(x),
            "mean_df2": np.mean(y),
            "mean_diff": np.mean(x) - np.mean(y),
            "p_value": p,
            "cohen_d": d
        })

    res_df = pd.DataFrame(results)
    res_df.sort_values("p_value", inplace=True)
    return res_df.reset_index(drop=True)

if __name__ == "__main__":
    hyperparams = {
        "merged_episodes": False,
        "bis_threshold": 70,
        "mac_threshold": 0.8,
        "min_episode_length": 20,
        "refractory_time": 5,
        "fixed_window_size": 20,
        "overlap": 0.0
    }

    analyze_center_of_mass_sets(hyperparams, 0.25, 0, 1)
