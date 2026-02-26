import pandas as pd
from collections import Counter
from pathlib import Path
from MachineLearning.IO.load_data import LoadData
from MachineLearning.IO.save_result import SaveResult

loader = LoadData()


def get_epoch_distribution_for_run(hyperparameters: dict, model_key: str, run_name: str, save_result=True):

    # Get splits
    relevant_splits = loader.return_related_fullpaths(hyperparameters, run_name, ["test_and_train_data", "splits"])

    # Get label mapping
    metadata = loader.load_run_data(hyperparameters, run_name, model_key)
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

def get_initial_epoch_distribution(hyperparameters: dict, save_result=True):
    faw_epochs = loader.load_faw_times_as_df(hyperparameters)
    awake_epochs = loader.load_awake_times_as_df(hyperparameters)

    faw_epoch_counts = Counter(faw_epochs["ResultID"])
    awake_epoch_counts = Counter(awake_epochs["ResultID"])

    rows = []
    for patient_id in faw_epoch_counts.keys() | awake_epoch_counts.keys():
        row = {"ResultID": patient_id, "faw_epochs": faw_epoch_counts.get(patient_id, 0),
               "awake_epochs": awake_epoch_counts.get(patient_id, 0)}
        rows.append(row)

    result_df = pd.DataFrame(rows).sort_values(by="ResultID").reset_index(drop=True)

    if save_result:
        from MachineLearning.IO.save_result import PathUtils
        a_b_c_d_dir = PathUtils.return_A_B_C_D_path("Dist", hyperparameters)
        x_y_name = PathUtils.return_X_Y_name(hyperparameters)
        folderpath = "D:\\Daten\\Other\\Initial_epoch_distribution\\"
        full_folderpath = Path(folderpath, a_b_c_d_dir, x_y_name)
        try:
            full_folderpath.mkdir()
        except FileExistsError:
            print(f"Creation of File with Path {full_folderpath} failed. It already exists")
        fullpath = Path(full_folderpath,"epoch_dist.csv")

        PathUtils.save_file_as_csv(result_df, fullpath)

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

    # epoch_dist_results = get_epoch_distribution_for_run(hyperparams, _model_key, _run_name)
    epoch_dist_results = get_initial_epoch_distribution(hyperparams, save_result=False)
    print(epoch_dist_results)
    print("Sums of epochs per type:\n")
    print(f"Faw_sum: {epoch_dist_results["faw_epochs"].sum()}\n")
    print(f"Awake_sum: {epoch_dist_results["awake_epochs"].sum()}\n")




