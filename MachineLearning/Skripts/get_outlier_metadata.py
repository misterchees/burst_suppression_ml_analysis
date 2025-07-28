from MachineLearning.Skripts.outlier_selection import select_multiple_outliers
from MachineLearning.IO.load_data import LoadData
from MachineLearning.IO.save_result import SaveResult
from MachineLearning.Utils.config_handler import load_config
import pandas as pd
from scipy.stats import ttest_ind, fisher_exact
import numpy as np

loader = LoadData()
saver = SaveResult()


def select_and_calculate_outliers(new_params, model_key, metadata_to_check, outlier_run_name, map_dict, save_result=True):
    numeric_cols = ["age", "asa", "height", "weight", "los_icu"]

    outlier_df = select_multiple_outliers(new_params, model_key, print_outliers=False,
                                          outlier_run_name=outlier_run_name)
    outlier_ids = outlier_df["group"].tolist()
    metadata_path = loader.return_csv_path_from_basedir("metadata")
    metadata_df = pd.read_csv(metadata_path).copy()
    metadata_df = convert_categorical_to_number(map_dict, metadata_df)

    # If numeric values are interpreted as strings -> convert to numeric
    for col in numeric_cols:
        metadata_df[col] = pd.to_numeric(metadata_df[col], errors="coerce")

    outlier_metadata_df, analysis_df = analyze_outlier_metadata(outlier_ids, metadata_df, metadata_to_check, map_dict)
    statistics_df = run_statistical_tests(outlier_metadata_df, metadata_df)

    if save_result:
        hyperparams = load_config("parameters_config.yaml")["current_params"]
        saver.save_metadata_analysis(
            outlier_metadata_df, model_key, hyperparams, "dataframe", "outliers_metadata", "slice", outlier_run_name
        )
        saver.save_metadata_analysis(
            analysis_df, model_key, hyperparams, "dataframe", "outliers_metadata", "analysis", outlier_run_name
        )
        saver.save_metadata_analysis(
            statistics_df, model_key, hyperparams, "dataframe", "outliers_metadata", "statistics", outlier_run_name
        )

    return outlier_metadata_df, analysis_df, statistics_df


def analyze_outlier_metadata(result_ids, metadata_df, metadata_to_check, map_dict):
    """
    Analyze metadata distribution for selected outlier ResultIDs vs. overall distribution.

    :param result_ids: List of ResultIDs that are considered outliers.
    :param metadata_df: Full metadata DataFrame including all ResultIDs.
    :param metadata_to_check: List of columns to analyze.
    :return: (outlier_metadata_df, analysis_df)
    """
    # Convert ids and metadata ids to string
    result_ids = [str(rid) for rid in result_ids]
    metadata_df["caseid"] = metadata_df["caseid"].astype(str)

    # 1. filter metadata for outlier
    outlier_metadata_df = metadata_df[metadata_df["caseid"].isin(result_ids)].copy()

    # 2. Extract only relevant columns
    outlier_metadata_df = outlier_metadata_df[["caseid"] + metadata_to_check].copy()
    full_metadata_df = metadata_df[["caseid"] + metadata_to_check].copy()

    # 4. Calculate comparisons of outliers with whole data
    analysis_rows = []
    for col in metadata_to_check:
        outlier_values = outlier_metadata_df[col].dropna()
        full_values = full_metadata_df[col].dropna()

        if col in map_dict:  # Calculations for binary values
            outlier_ratio = outlier_values.mean()
            full_ratio = full_values.mean()
            diff = outlier_ratio - full_ratio
            analysis_rows.append({
                "variable": col,
                "type": "binary",
                "outlier_mean": outlier_ratio,
                "full_mean": full_ratio,
                "difference": diff
            })
        else:  # Calculation for numerical values
            outlier_mean = outlier_values.mean()
            full_mean = full_values.mean()
            diff = outlier_mean - full_mean
            analysis_rows.append({
                "variable": col,
                "type": "numeric",
                "outlier_mean": outlier_mean,
                "full_mean": full_mean,
                "difference": diff
            })

    analysis_df = pd.DataFrame(analysis_rows)

    return outlier_metadata_df, analysis_df


def run_statistical_tests(outlier_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform statistical tests to compare outlier vs full dataset.

    :param outlier_df: DataFrame of outlier patients.
    :param full_df: DataFrame of all patients.
    :return: DataFrame with statistical test results.
    """
    results = []

    for col in outlier_df.columns:
        if col == "caseid":
            continue

        outlier_vals = outlier_df[col].dropna()
        full_vals = full_df[col].dropna()

        # Determine type of variable
        unique_vals = full_vals.dropna().unique()
        is_binary = sorted(unique_vals) in ([0, 1], [1, 0])

        if is_binary:
            # Fisher’s Exact Test
            try:
                outlier_counts = np.bincount(outlier_vals.astype(int), minlength=2)
                full_counts = np.bincount(full_vals.astype(int), minlength=2)

                contingency_table = [
                    [outlier_counts[1], outlier_counts[0]],
                    [full_counts[1], full_counts[0]]
                ]

                _, p_value = fisher_exact(contingency_table)
                test_type = "Fisher's Exact"
            except Exception as e:
                p_value = np.nan
                test_type = f"Fisher error: {e}"

        else:
            # Welch's t-test (independent samples, unequal variance)
            try:
                _, p_value = ttest_ind(outlier_vals, full_vals, equal_var=False)
                test_type = "Welch's t-test"
            except Exception as e:
                p_value = np.nan
                test_type = f"T-test error: {e}"

        results.append({
            "variable": col,
            "type": "binary" if is_binary else "numeric",
            "outlier_mean": round(np.mean(outlier_vals), 4),
            "full_mean": round(np.mean(full_vals), 4),
            "difference": round(np.mean(outlier_vals) - np.mean(full_vals), 4),
            "test": test_type,
            "p_value": round(p_value, 4)
        })

    return pd.DataFrame(results)


def convert_categorical_to_number(categorical_to_number_dict: dict, df_to_convert: pd.DataFrame) -> pd.DataFrame:
    for col, mapping in categorical_to_number_dict.items():
        if col in df_to_convert.columns:
            df_to_convert[col] = df_to_convert[col].map(mapping)

    return df_to_convert


if __name__ == "__main__":
    # asa: American Society of Anaesthesiologists Score
    # emop: Emergency operation (y or n)
    # preop_htn: Preoperative hypertension (y or n)
    # preop_dm: Preoperative diabetes (y or n)
    # los_icu: Postoperative length of ICU stay in days

    _metadata_to_check = ["age", "sex", "asa", "emop", "preop_htn", "preop_dm", "height", "weight", "death_inhosp", "los_icu"]
    _new_params = {
        "current_params": {
            "merged_episodes": False,
            "bis_threshold": 70,
            "mac_threshold": 0.8,
            "min_episode_length": 20,
            "refractory_time": 5,
            "fixed_window_size": 20,
            "overlap": 0.0
        }
    }

    _map_dict = {
        "sex": {"M": 0, "F": 1},
        "emop": {"N": 0, "Y": 1},
        "preop_htn": {"N": 0, "Y": 1},
        "preop_dm": {"N": 0, "Y": 1},
        "death_inhosp": {"N": 0, "Y": 1}
    }

    _model_key = "svm"
    _outlier_run_name = "test_run_0"

    _outlier_metadata_df, _analysis_df, _statistics_df = select_and_calculate_outliers(
        _new_params, _model_key, _metadata_to_check, _outlier_run_name, _map_dict
    )
    print(_outlier_metadata_df)
    print(_analysis_df)
    print(_statistics_df)
