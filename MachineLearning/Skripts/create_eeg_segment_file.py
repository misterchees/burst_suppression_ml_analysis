import numpy as np
import pandas as pd

from MachineLearning.Utils.feature_utils import FeatureUtils


def create_eeg_segment_file(epoch_type_0: str, epoch_type_1: str, parameters: dict):
    type_0_eeg_tuples = FeatureUtils.return_eeg_epochs(epoch_type_0, parameters)
    type_1_eeg_tuples = FeatureUtils.return_eeg_epochs(epoch_type_1, parameters)

    segment_df = create_epoch_dataframe(type_0_eeg_tuples, type_1_eeg_tuples)

    filtered_df = remove_empty_eeg_segments(segment_df)

    # Convert EEG series to dataframe
    eeg_array = np.stack(filtered_df['eeg'].values)  # (n_samples, 896)
    eeg_df = pd.DataFrame(eeg_array, columns=[f"eeg_{i}" for i in range(896)])

    # Kombinieren mit result_id und label
    df_combined = pd.concat([filtered_df[['patient_id']].reset_index(drop=True),
                             eeg_df,
                             filtered_df[['label']].reset_index(drop=True)], axis=1)

    min_episode_length = parameters["min_episode_length"]
    path_to_folder = f"D:\\Daten\\Other\\minlength_{min_episode_length}_fixlength_7_overlap_075"
    df_combined.to_pickle(f"{path_to_folder}.pkl")
    df_combined.to_feather(f"{path_to_folder}.feather")
    df_combined.to_parquet(f"{path_to_folder}.parquet")


def create_epoch_dataframe(label_0_list: list[tuple], label_1_list: list[tuple]) -> pd.DataFrame:
    """
    Creates a DataFrame from two lists of EEG epoch tuples with labels 0 and 1.

    :param label_0_list: List of 5-tuples with label 0 (start, end, result_id, fs, eeg_data)
    :param label_1_list: List of 5-tuples with label 1 (start, end, result_id, fs, eeg_data)
    :return: pd.DataFrame with columns: ['patient_id', 'eeg', 'label']
    """

    def convert_list_to_df(data_list, label):
        return pd.DataFrame([
            {
                "patient_id": tup[2],
                "eeg": tup[4],
                "label": label
            }
            for tup in data_list
        ])

    df_0 = convert_list_to_df(label_0_list, label=0)
    df_1 = convert_list_to_df(label_1_list, label=1)

    return pd.concat([df_0, df_1], ignore_index=True)


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


if __name__ == '__main__':
    hyperparams = {
        "merged_episodes": False,
        "bis_threshold": 70,
        "mac_threshold": 0.8,
        "min_episode_length": 10,
        "refractory_time": 5,
        "fixed_window_size": 7,
        "overlap": 0.75
    }

    create_eeg_segment_file("faw", "awake", hyperparams)
