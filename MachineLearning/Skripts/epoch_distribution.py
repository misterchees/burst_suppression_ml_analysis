
import os
import pandas as pd
from collections import Counter
from MachineLearning.IO.load_data import LoadData, PathUtils
from MachineLearning.IO.save_result import SaveResult

loader = LoadData()


def get_epoch_distribution_for_run(hyperparameters: dict, model_key: str, run_name: str, save_result=True):

    # Get splits
    splits_folderpath = loader.return_all_parameter_fullpath(
        hyperparameters, False, False, ["test_and_train_data", "splits"], run_name
    )
    split_fullpaths, _ = PathUtils.list_files_in_folder(splits_folderpath, ".csv", fullpaths=True)

    # Filter out all files that are not splits
    relevant_splits = [
        path for path in split_fullpaths
        if os.path.basename(path).endswith(("train_split.csv", "test_split.csv"))
    ]
    # Validation for split folder
    if not relevant_splits:
        raise FileNotFoundError(f"No valid splits found in folder {splits_folderpath} for run_name='{run_name}'")

    # --- Get label mapping ---
    metadata_folderpath = loader.return_all_parameter_fullpath(
        hyperparameters, False, False, ["run_metadata", model_key]
    )
    metadata_fullpaths, _ = PathUtils.list_files_in_folder(metadata_folderpath, ".json", fullpaths=True)

    # Search for run metadata
    matching_metadata_path = next(
        (path for path in metadata_fullpaths if os.path.basename(path) == f"{run_name}.json"),
        None
    )
    if not matching_metadata_path:
        raise FileNotFoundError(f"No matching file for run_name='{run_name}' found in folder='{metadata_folderpath}'.")

    # Get label mapping
    metadata = PathUtils.load_json(matching_metadata_path)
    epoch_type_list = metadata.get("epoch_type", [])
    label_mapping = {i: label_name for i, label_name in enumerate(epoch_type_list)}

    # Read all split data
    all_splits = []

    for split_path in relevant_splits:
        df = pd.read_csv(split_path)
        all_splits.append(df)

    # Combine splits and drop duplicates
    combined_df = pd.concat(all_splits, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["ResultID", "Start", "End"])

    # Count splits in default dict
    patient_label_counts = {}
    for _, row in combined_df.iterrows():
        patient_id = row["ResultID"]
        label = int(row["label"])
        label_name = label_mapping.get(label, f"Label_{label}")  # Label according to label dict
        if patient_id not in patient_label_counts:
            patient_label_counts[patient_id] = Counter()
        patient_label_counts[patient_id][label_name] += 1  # Dict -> {patient: {label: label count}}

    # Create overview
    all_label_names = sorted(set(label for counts in patient_label_counts.values() for label in counts))
    rows = []
    for patient_id, counts in patient_label_counts.items():
        row = {"ResultID": patient_id}
        for label_name in all_label_names:
            row[label_name] = counts.get(label_name, 0)
        rows.append(row)

    result_df = pd.DataFrame(rows).sort_values(by="ResultID").reset_index(drop=True)

    if save_result:
        saver = SaveResult()
        saver.save_metadata_analysis(result_df, model_key, hyperparameters, "dataframe", "Summary","epoch_distribution", run_name)

    return result_df


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
    _model_key = "svm"
    _run_name = "test_run_0"

    epoch_dist_results = get_epoch_distribution_for_run(hyperparams, _model_key, _run_name)
    print(epoch_dist_results)



