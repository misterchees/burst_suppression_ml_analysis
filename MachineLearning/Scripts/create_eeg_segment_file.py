import numpy as np
import pandas as pd

from MachineLearning.Utils.feature_utils import FeatureUtils
from MachineLearning.IO.load_data import LoadData
from MachineLearning.Utils.path_manager import PathManager
from pathlib import Path
from typing import List, Any


def create_eeg_segment_file(epoch_type_0: str, epoch_type_1: str, parameters: dict, _run_name: str = None):
    if _run_name is not None:
        ids_from_run = get_ids_from_run(_run_name)
    else:
        ids_from_run = None

    type_0_eeg_tuples = FeatureUtils.return_eeg_epochs(epoch_type_0, parameters, allowed_ids=ids_from_run)
    type_1_eeg_tuples = FeatureUtils.return_eeg_epochs(epoch_type_1, parameters, allowed_ids=ids_from_run)

    segment_df = create_epoch_dataframes(type_0_eeg_tuples, type_1_eeg_tuples)

    filtered_df = remove_empty_eeg_segments(segment_df)

    # Convert EEG series to dataframe
    eeg_array = np.stack(filtered_df['eeg'].values)  # (n_samples, 896)
    eeg_df = pd.DataFrame(eeg_array, columns=[f"eeg_{i}" for i in range(896)])

    # Combine with result_id und label
    df_combined = pd.concat([filtered_df[['patient_id']].reset_index(drop=True),
                             eeg_df,
                             filtered_df[['label']].reset_index(drop=True)], axis=1)

    min_episode_length = parameters["min_episode_length"]
    path_to_folder = f"D:\\Daten\\Other\\minlength_{min_episode_length}_fixlength_7_overlap_075_no_outliers"
    df_combined.to_pickle(f"{path_to_folder}.pkl")
    df_combined.to_feather(f"{path_to_folder}.feather")
    df_combined.to_parquet(f"{path_to_folder}.parquet")


def create_epoch_dataframes(label_0_list: list[tuple], label_1_list: list[tuple]) -> pd.DataFrame:
    """
    Creates a DataFrame from two lists of EEG epoch tuples with labels 0 and 1.

    :param label_0_list: List of 5-tuples with label 0 (start, end, result_id, fs, eeg_data)
    :param label_1_list: List of 5-tuples with label 1 (start, end, result_id, fs, eeg_data)
    :return: pd.DataFrame with columns: ['patient_id', 'eeg', 'label']
    """

    df_0 = convert_list_to_df(label_0_list, label=0)
    df_1 = convert_list_to_df(label_1_list, label=1)

    return pd.concat([df_0, df_1], ignore_index=True)


def convert_list_to_df(data_list: List[tuple], label: Any) -> pd.DataFrame:
    """
    Converts a list of tuples into a DataFrame with patient_id, eeg, and label columns.
    Filters by patient_ids if provided, and sorts the result by patient_id.

    :param data_list: List of tuples, where tup[2] is patient_id and tup[4] is EEG data
    :param label: Label value to assign to all rows
    :return: Filtered and sorted DataFrame
    """
    df = pd.DataFrame([
        {
            "patient_id": tup[2],
            "eeg": tup[4],
            "label": label
        }
        for tup in data_list
    ])

    df = df.sort_values(by="patient_id").reset_index(drop=True)
    return df


def remove_empty_eeg_segments(df: pd.DataFrame, eeg_col: str = "eeg") -> pd.DataFrame:
    """
    Removes rows from the DataFrame where the EEG segment is empty.

    :param df: Input DataFrame with an EEG column.
    :param eeg_col: Name of the column containing EEG segments.
    :return: Filtered DataFrame with non-empty EEG segments only.
    """
    initial_count = len(df)

    # Boolean mask: only keep rows where EEG is a list or array and not empty
    mask = df[eeg_col].apply(lambda x: isinstance(x, (list, np.ndarray)) and len(x) == 896)
    filtered_df = df[mask].reset_index(drop=True)

    removed_count = initial_count - len(filtered_df)

    print(f"[Info] Removed {removed_count} empty EEG segments. Remaining: {len(filtered_df)}")

    return filtered_df


def get_ids_from_run(_run_name: str) -> list:
    pm = PathManager()
    # All run metadata are currently in a folder with this coded information
    metadata_params = {
        "merged_episodes": False,
        "bis_threshold": 70,
        "mac_threshold": 0.8,
        "min_episode_length": 20,
        "refractory_time": 5,
        "fixed_window_size": 20,
        "overlap": 0.0
    }
    run_folderpath = pm.get_complex_ml_path(metadata_params, ["run_metadata", "svm"], False, False)
    fullpath = run_folderpath / f"{_run_name}.json"

    run_metadata = LoadData.load_json(fullpath)
    used_patient_ids = run_metadata["final_patient_ids"]
    print(f"Patient IDs that will be used: {used_patient_ids}")
    return used_patient_ids


if __name__ == '__main__':
    hyperparams = {
        "merged_episodes": False,
        "bis_threshold": 70,
        "mac_threshold": 0.8,
        "min_episode_length": 20,
        "refractory_time": 5,
        "fixed_window_size": 7,
        "overlap": 0.75
    }
    run_name = "norm2_z_score_0"

    create_eeg_segment_file("faw", "awake", hyperparams, run_name)
