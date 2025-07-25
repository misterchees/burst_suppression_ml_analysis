from MachineLearning.Skripts.outlier_selection import select_multiple_outliers
from MachineLearning.IO.load_data import LoadData
from MachineLearning.IO.save_result import SaveResult
from MachineLearning.Utils.config_handler import load_config
import pandas as pd

loader = LoadData()
saver = SaveResult()


def select_and_calculate_outliers(new_params, model_key, metadata_to_check, outlier_run_name, save_result=True):
    numeric_cols = ["age", "asa"]

    outlier_df = select_multiple_outliers(new_params, model_key, print_outliers=False,
                                          outlier_run_name=outlier_run_name)
    outlier_ids = outlier_df["group"].tolist()
    metadata_path = loader.return_csv_path_from_basedir("metadata")
    metadata_df = pd.read_csv(metadata_path)

    # If numeric values are interpreted as strings -> convert to numeric
    for col in numeric_cols:
        metadata_df[col] = pd.to_numeric(metadata_df[col], errors="coerce")

    outlier_metadata_df, analysis_df = analyze_outlier_metadata(outlier_ids, metadata_df, metadata_to_check)

    if save_result:
        hyperparams = load_config("parameters_config.yaml")["current_params"]
        saver.save_metadata_analysis(
            outlier_metadata_df, model_key, hyperparams, "dataframe", "outliers_metadata", "slice"
        )
        saver.save_metadata_analysis(
            analysis_df, model_key, hyperparams, "dataframe", "outliers_metadata", "analysis"
        )

    return outlier_metadata_df, analysis_df


def analyze_outlier_metadata(result_ids, metadata_df, metadata_to_check):
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

    # 3. Code kategorical values
    map_dict = {
        "sex": {"M": 0, "F": 1},
        "emop": {"N": 0, "Y": 1},
        "preop_htn": {"N": 0, "Y": 1},
        "preop_dm": {"N": 0, "Y": 1},
    }
    for col, mapping in map_dict.items():
        if col in outlier_metadata_df.columns:
            outlier_metadata_df[col] = outlier_metadata_df[col].map(mapping)
            full_metadata_df[col] = full_metadata_df[col].map(mapping)

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


if __name__ == "__main__":
    # asa: American Society of Anaesthesiologists Score
    # emop: Emergency operation (y or n)
    # preop_htn: Preoperative hypertension (y or n)
    # preop_dm: Preoperative diabetes (y or n)

    _metadata_to_check = ["age", "sex", "asa", "emop", "preop_htn", "preop_dm"]
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
    _model_key = "svm"
    _outlier_run_name = "test_run_0"

    _outlier_metadata_df, _analysis_df = select_and_calculate_outliers(
        _new_params, _model_key, _metadata_to_check, _outlier_run_name
    )
    print(_outlier_metadata_df)
    print(_analysis_df)
