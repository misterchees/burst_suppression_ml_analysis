import os
import numpy as np
import pandas as pd
from MachineLearning.IO.io_core import IOCore
from MachineLearning.Utils.path_utils import PathUtils


class SaveResult(IOCore):
    def __init__(self):
        super().__init__()

    def save_faw_psd(self, frequencies: np.ndarray, power: np.ndarray,
                     parameters: dict, start: int, end: int, result_id: int):
        """
        Saves PSD data of EEG in a directory specified by parameters with a name specified by start, end and result_id.
        :param frequencies: Frequencies from the PSD
        :param power: Power of the Frequencies from the PSD
        :param parameters: A dictionary with all episode parameters from the project
        :param start: Start time of episode, from which the PSD was calculated
        :param end: End time of episode, from which the PSD was calculated.
        :param result_id: Patient ID
        """

        # assemble path to directory
        psd_dir_fullpath = self.return_all_parameter_fullpath(parameters, False, True, "features", "psds")

        # Save file in directory
        self.save_psd_in_given_directory(frequencies, power, start, end, result_id, psd_dir_fullpath)

    def save_awake_psd(self, frequencies: np.ndarray, power: np.ndarray,
                       parameters: dict, start: int, end: int, result_id: int):
        """
        Saves PSD data of EEG in a directory specified by parameters with a name specified by start, end and result_id.
        :param frequencies: Frequencies from the PSD
        :param power: Power of the Frequencies from the PSD
        :param parameters: A dictionary with all episode parameters from the project
        :param start: Start time of episode, from which the PSD was calculated
        :param end: End time of episode, from which the PSD was calculated.
        :param result_id: Patient ID
        """

        # assemble path to directory
        psd_dir_fullpath = self.return_no_parameters_fullpath(parameters, "awake", False, "features", "psds")

        # make sure directory exists
        os.makedirs(psd_dir_fullpath, exist_ok=True)

        self.save_psd_in_given_directory(frequencies, power, start, end, result_id, psd_dir_fullpath)

    def save_normal_an_psd(self, frequencies: np.ndarray, power: np.ndarray,
                           parameters: dict, start: int, end: int, result_id: int):
        """
        Saves PSD data of EEG in a directory specified by parameters with a name specified by start, end and result_id.
        :param frequencies: Frequencies from the PSD
        :param power: Power of the Frequencies from the PSD
        :param parameters: A dictionary with all episode parameters from the project
        :param start: Start time of episode, from which the PSD was calculated
        :param end: End time of episode, from which the PSD was calculated.
        :param result_id: Patient ID
        """

        # assemble path to directory
        psd_dir_fullpath = self.return_no_parameters_fullpath(parameters, "normal_an", False, "features", "psds")

        # make sure directory exists
        os.makedirs(psd_dir_fullpath, exist_ok=True)

        self.save_psd_in_given_directory(frequencies, power, start, end, result_id, psd_dir_fullpath)

    def save_psd_in_given_directory(self, frequencies: np.ndarray, power: np.ndarray,
                                    start: int, end: int, result_id: int, psd_dir_fullpath: str):
        """
        Saves PSD data of EEG in a given directory with a name specified by start, end and result_id.
        :param frequencies: Frequencies from the PSD
        :param power: Power of the Frequencies from the PSD
        :param start: Start time of episode, from which the PSD was calculated
        :param end: End time of episode, from which the PSD was calculated.
        :param result_id: Patient ID
        :param psd_dir_fullpath: Directory where the PSD will be saved.
        """
        # create dataframe from PSD data
        psd_cols = self.data_names["psd_files"]
        psd_df = pd.DataFrame({
            psd_cols["psd_freq_col"]: frequencies,
            psd_cols["psd_power_col"]: power
        })

        # create fullpath with PSD name to save data
        psd_filename = f"PSD_{start}_{end}_{result_id}.csv"
        fullpath = PathUtils.return_anypath(psd_dir_fullpath, psd_filename)

        # save psd
        psd_df.to_csv(fullpath, index=False)

        print(f"Single episode PSD saved: {fullpath}")

    def save_wholeEEG_psd(self, frequencies: np.ndarray, power: np.ndarray, filtered: bool, result_id: int):
        """
        Saves whole EEG PSDs in the PSD directory
        :param frequencies: Frequencies from the PSD
        :param power: Power of the Frequencies from the PSD
        :param filtered: Metadata if PSD is from a filtered EEG.
        :param result_id: Patient ID corresponding to EEG.
        """
        # assemble path to directory
        psd_dir = self.return_folder_path("features", "psds")

        if filtered:
            filter_prefix = "filtered"
        else:
            filter_prefix = "raw"

        psd_filename = f"PSD_{filter_prefix}_whole_EEG_{result_id}.csv"
        whole_eeg_subdir = PathUtils.return_anypath(psd_dir, "whole_EEG_PSD", filter_prefix)

        # make sure directory exists
        os.makedirs(whole_eeg_subdir, exist_ok=True)

        # create dataframe from PSD data
        psd_cols = self.data_names["psd_files"]
        psd_df = pd.DataFrame({
            psd_cols["psd_freq_col"]: frequencies,
            psd_cols["psd_power_col"]: power
        })

        # create fullpath with PSD name to save data
        fullpath = PathUtils.return_anypath(whole_eeg_subdir, psd_filename)

        # save psd
        psd_df.to_csv(fullpath, index=False)

        print(f"Single episode PSD saved: {fullpath}")

    def save_feature_summary_episode(self, results: list, feature_key: str, parameters: dict, episode_type: str):
        result_df = pd.DataFrame(results)
        fullpath = self.return_file_fullpath(parameters, True, True, episode_type,
                                             "features", feature_key)

        result_df.to_csv(fullpath, index=False)

    def save_filtered_eeg(self, filtered_eeg: np.ndarray, fs: int, result_id: int):
        """
        Saves a filtered EEG as a <result_id>.csv. Assuming the EEG has only 2 channels, these
        will be the columns of the csv-file, and a third column for the fs (sampling frequency)
        :param filtered_eeg: An array of filtered EEG (assuming 2 columns i.e. channels)
        :param fs: The sampling frequency of the EEG
        :param result_id: The patient ID. Will be part of the saved file -> <result_id>.csv
        """
        channels = self.data_names["eeg_files"]["eeg_channels"]
        df = pd.DataFrame(filtered_eeg, columns=channels)
        filtered_eeg_subdir = self.return_folder_path("filtered_data")
        os.makedirs(os.path.dirname(filtered_eeg_subdir), exist_ok=True)
        fullpath = PathUtils.return_anypath(filtered_eeg_subdir, f"{result_id}.csv")

        # Write fs as header line, then the rounded data
        with open(fullpath, "w", newline='') as f:
            f.write(f"# fs = {fs}\n")
            df.to_csv(f, index=False, float_format="%.4f")

        print(f"Successfully saved filtered EEG to {fullpath}")

    def save_combined_features(self, parameters: dict, merged_df: pd.DataFrame, epoch_type: str):
        """
        Saves a dataframe of combined features to a specified folder.
        :param parameters: parameters for the features -> define the subfolder names
        :param merged_df: Dataframe of all features combined to save
        :param epoch_type: Defines the epochs from which the features are calculated.
        """
        fullpath = self.return_file_fullpath(parameters, True, True, epoch_type, "test_and_train_data", "feature_sets")
        merged_df.to_csv(fullpath, index=False)
        print(f"Combined feature set saved to: {fullpath}")

    def save_single_split(self, parameters: dict, train_test_tuple: tuple):
        """
        Saves created train and test splits to a folder defined by current parameters.
        :param parameters: Determines saving folder.
        :param train_test_tuple: Tuple containing the train and test split. Order is (train, test)
        """
        train_fullpath = self.return_single_split_folder_fullpath(parameters, "train")
        test_fullpath = self.return_single_split_folder_fullpath(parameters, "test")

        train_df, test_df = train_test_tuple
        print(f"Saving train data to {train_fullpath}\n Saving test data to {test_fullpath}")
        train_df.to_csv(train_fullpath, index=False)
        test_df.to_csv(test_fullpath, index=False)

        print("Split saving successful")

    def save_cv_splits_to_csv(self, parameters: dict, split_object):
        """
        Saves for every fold train and test data as csv file.

        :param parameters: Determines saving folder.
        :param split_object: A tuple that contains X, y, splits
        - DataFrame with Features (without label column)
        - Array with label values
        - List of (train_idx, test_idx)-Tupels
        """
        # Unpack Split Object
        X, y, splits = split_object

        # Add labels to X
        X_labeled = X.copy()
        X_labeled["label"] = y

        total_folds = len(splits)

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            train_df = X_labeled.iloc[train_idx]
            test_df = X_labeled.iloc[test_idx]

            train_path = self.return_folded_split_folder_fullpath(parameters, "train", fold_idx+1, total_folds)
            test_path = self.return_folded_split_folder_fullpath(parameters, "test", fold_idx+1, total_folds)

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            print(f"Saved fold {fold_idx + 1}/{total_folds} to:")
            print(f" - {train_path}")
            print(f" - {test_path}")

    def save_model(self, model, model_key: str, parameters: dict):
        """
        Saves a model to a specified folder.
        :param model: The model to be saved.
        :param model_key: Key of the model (name)
        :param parameters: Parameters for the model -> define the subfolder names.
        """

        from joblib import dump
        full_folder_path = self.return_all_parameter_fullpath(parameters, False, True, "models", model_key)
        model_file = f"{model_key}.joblib"
        fullpath = PathUtils.return_anypath(full_folder_path, model_file)
        dump(model, fullpath)


