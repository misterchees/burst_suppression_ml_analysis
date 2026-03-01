"""This module contains the SaveResult class"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from MachineLearning.IO.io_core import IOCore, List
from MachineLearning.Utils.path_manager import PathManager
from MachineLearning.Utils.file_data_utils import FileDataUtils


class SaveResult(IOCore):
    """This class handles the saving of any data in this project"""
    def __init__(self):
        """Initializes the class with configs from IOCore superclass."""
        super().__init__()
        self.pm = PathManager()

    def save_faw_psd(self, frequencies: np.ndarray, power: np.ndarray,
                     parameters: dict, start: int, end: int, result_id: int):
        """
        Saves PSD data of EEG in a directory specified by parameters with a name specified by start, end, and result_id.
        :param frequencies: Frequencies from the PSD
        :param power: Power of the Frequencies from the PSD
        :param parameters: A dictionary with all episode parameters from the project
        :param start: Start time of episode, from which the PSD was calculated
        :param end: End time of episode, from which the PSD was calculated.
        :param result_id: Patient ID
        """

        # Assemble path to directory
        psd_dir_fullpath = self.pm.get_complex_ml_path(parameters, ["features", "psds"], True, False)

        # Save file in directory
        self.save_psd_in_given_directory(frequencies, power, start, end, result_id, psd_dir_fullpath)

    def save_awake_psd(self, frequencies: np.ndarray, power: np.ndarray,
                       parameters: dict, start: int, end: int, result_id: int):
        """
        Saves PSD data of EEG in a directory specified by parameters with a name specified by start, end, and result_id.
        :param frequencies: Frequencies from the PSD
        :param power: Power of the Frequencies from the PSD
        :param parameters: A dictionary with all episode parameters from the project
        :param start: Start time of episode, from which the PSD was calculated
        :param end: End time of episode, from which the PSD was calculated.
        :param result_id: Patient ID
        """

        # Assemble path to directory
        psd_dir_fullpath = self.pm.get_simple_episode_path(parameters, "awake", ["features", "psds"], False )

        # Make sure directory exists
        psd_dir_fullpath.mkdir(exist_ok=True)

        self.save_psd_in_given_directory(frequencies, power, start, end, result_id, psd_dir_fullpath)

    def save_normal_an_psd(self, frequencies: np.ndarray, power: np.ndarray,
                           parameters: dict, start: int, end: int, result_id: int):
        """
        Saves PSD data of EEG in a directory specified by parameters with a name specified by start, end, and result_id.
        :param frequencies: Frequencies from the PSD
        :param power: Power of the Frequencies from the PSD
        :param parameters: A dictionary with all episode parameters from the project
        :param start: Start time of episode, from which the PSD was calculated
        :param end: End time of episode, from which the PSD was calculated.
        :param result_id: Patient ID
        """

        # Assemble path to directory
        psd_dir_fullpath = self.pm.get_simple_episode_path(parameters, "normal_an", ["features", "psds"], False )

        # Make sure directory exists
        psd_dir_fullpath.mkdir(exist_ok=True)

        self.save_psd_in_given_directory(frequencies, power, start, end, result_id, psd_dir_fullpath)

    def save_psd_in_given_directory(self, frequencies: np.ndarray, power: np.ndarray,
                                    start: int, end: int, result_id: int, psd_dir_fullpath: Path):
        """
        Saves PSD data of EEG in a given directory with a name specified by start, end, and result_id.
        :param frequencies: Frequencies from the PSD
        :param power: Power of the Frequencies from the PSD
        :param start: Start time of the episode, from which the PSD was calculated
        :param end: End time of the episode, from which the PSD was calculated.
        :param result_id: Patient ID
        :param psd_dir_fullpath: Directory where the PSD will be saved.
        """
        # Validate simple values
        if start is None or end is None or result_id is None:
            raise ValueError("start, end and result_id must be values")

        # Create dataframe from PSD data
        psd_cols = self.data_names["psd_files"]
        psd_df = pd.DataFrame({
            psd_cols["psd_freq_col"]: frequencies,
            psd_cols["psd_power_col"]: power
        })

        # Create a fullpath with the PSD name to save data
        psd_filename = f"PSD_{start}_{end}_{result_id}.csv"
        fullpath = Path(psd_dir_fullpath, psd_filename)

        # Save psd
        psd_df.to_csv(fullpath, index=False)

        print(f"Single episode PSD saved: {fullpath}")

    def save_complete_eeg_psd(self, frequencies: np.ndarray, power: np.ndarray, filtered: bool, result_id: int):
        """
        Saves whole EEG PSDs in the PSD directory.
        :param frequencies: Frequencies from the PSD
        :param power: Power of the Frequencies from the PSD
        :param filtered: Metadata if PSD is from a filtered EEG.
        :param result_id: Patient ID corresponding to EEG.
        """
        # Assemble the path to the directory
        psd_dir = self.pm.get_path("features", "psds")

        if filtered:
            filter_prefix = "filtered"
        else:
            filter_prefix = "raw"

        psd_filename = f"PSD_{filter_prefix}_whole_EEG_{result_id}.csv"
        whole_eeg_subdir = Path(psd_dir, "whole_EEG_PSD", filter_prefix)

        # Make sure the directory exists
        whole_eeg_subdir.mkdir(exist_ok=True)

        # Create a dataframe from the PSD data
        psd_cols = self.data_names["psd_files"]
        psd_df = pd.DataFrame({
            psd_cols["psd_freq_col"]: frequencies,
            psd_cols["psd_power_col"]: power
        })

        # Create a fullpath with the PSD name to save data
        fullpath = Path(whole_eeg_subdir, psd_filename)

        # Save psd
        psd_df.to_csv(fullpath, index=False)

        print(f"Single episode PSD saved: {fullpath}")

    def save_feature_summary_episode(self, results: list, feature_key: str, parameters: dict, episode_type: str):
        """
        This function saves the calculated results of one feature in a directory depending on given parameters.
        :param results: A list of calculation results for a given feature.
        :param feature_key: The key for the feature.
        :param parameters: Parameters of the run -> determine the pathfile.
        :param episode_type: Type of the episode calculated. Valid options are 'normal_an', 'faw' and 'awake'
        :return: None
        """
        result_df = pd.DataFrame(results)
        fullpath = self.pm.resolve_episode_path(
            parameters, episode_type, ["features", feature_key], True, True
        )

        result_df.to_csv(fullpath, index=False)

    def save_eeg_track(self, eeg_track: np.ndarray, fs: int, result_id: int, folder_keys: List[str]):
        """
        Saves an EEG as a <result_id>.csv. Assuming the EEG has only 2 channels, these
        will be the columns of the csv-file, and the first row contains the fs (sampling frequency)
        :param eeg_track: An EEG track as an array (assuming 2 columns = channels)
        :param fs: The sampling frequency of the EEG
        :param result_id: The patient ID. Will be part of the saved file -> <result_id>.csv
        :param folder_keys: The keys to the folder in which the file should be saved.
        :return: None
        """
        # Get the number of each possible channel specified in config
        channels = self.data_names["eeg_files"]["eeg_channels"]
        df = pd.DataFrame(eeg_track, columns=channels)
        eeg_subdir = self.pm.get_path(*folder_keys) # Assemble a path to the dir of the file
        eeg_subdir.mkdir(exist_ok=True) # Make sure folders in the path exist
        fullpath = Path(eeg_subdir, f"{result_id}.csv")

        # Write fs as the header line, then the rounded data
        with open(fullpath, "w", newline='') as f:
            f.write(f"# fs = {fs}\n")
            df.to_csv(f, index=False, float_format="%.4f")

        print(f"Successfully saved EEG to {fullpath}")

    def save_combined_features(self, parameters: dict, merged_df: pd.DataFrame, epoch_type: str):
        """
        Saves a dataframe of combined features to a specified folder.
        :param parameters: Parameters for the features -> define the subfolder names
        :param merged_df: Dataframe of all features combined to save
        :param epoch_type: Defines the epochs from which the features are calculated
        """
        fullpath = self.pm.resolve_episode_path(
            parameters, epoch_type, ["test_and_train_data", "feature_sets"], True, True
        )
        merged_df.to_csv(fullpath, index=False)
        print(f"Combined feature set saved to: {fullpath}")

    def save_single_split(self, parameters: dict, train_test_tuple: tuple):
        """
        Saves a created train and test split to a folder defined by current parameters.
        :param parameters: Determines saving folder.
        :param train_test_tuple: Tuple containing the train and test split. Order is (train, test)
        """
        # Assemble paths where splits will be saved
        train_fullpath = self.return_single_split_folder_fullpath(parameters, "train")
        test_fullpath = self.return_single_split_folder_fullpath(parameters, "test")

        # Extract train and test data
        train_df, test_df = train_test_tuple
        print(f"Saving train data to {train_fullpath}\n Saving test data to {test_fullpath}")

        # Save data as CSV
        train_df.to_csv(train_fullpath, index=False)
        test_df.to_csv(test_fullpath, index=False)

        print("Split saving successful")

    def save_cv_splits_to_csv(self, parameters: dict, split_object):
        """
        Saves cross-validated splits. Every train and test data from each fold is saved separately.

        :param parameters: Parameters of the run -> determine saving folder.
        :param split_object: A tuple that contains (X, y, splits) where X is a dataframe with features without
         label column, y is the array with label values, and splits is a list of (train_idx, test_idx)-Tupels
        """
        # Unpack Split Object
        X, y, splits = split_object

        # Add labels to X
        X_labeled = X.copy()
        X_labeled["label"] = y

        total_folds = len(splits)

        # Iterate through each fold giving it a number in the process
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            train_df = X_labeled.iloc[train_idx]
            test_df = X_labeled.iloc[test_idx]

            # Create a path for each train and test data instance and save data accordingly
            train_path = self.return_folded_split_folder_fullpath(parameters, "train", fold_idx + 1, total_folds)
            test_path = self.return_folded_split_folder_fullpath(parameters, "test", fold_idx + 1, total_folds)
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            print(f"Saved fold {fold_idx + 1}/{total_folds} to:")
            print(f" - {train_path}")
            print(f" - {test_path}")

    def save_model(self, model, model_key: str, parameters: dict):
        """
        Saves a model to a specified folder via joblib.
        :param model: The model to be saved.
        :param model_key: Key of the model (Analogous to its name).
        :param parameters: Parameters for the model -> define the subfolder names.
        """

        from joblib import dump
        full_folder_path = self.pm.get_complex_ml_path(parameters, ["models", model_key], False, True)
        model_filename = f"{model_key}.joblib"
        fullpath = Path(full_folder_path, model_filename)
        dump(model, fullpath)

    def save_ml_result(self, result_data, model_key: str, parameters: dict,
                       file_type: str, file_prefix: str = "", file_suffix: str = ""):
        """
        Save machine learning result data to a specified file format and location.

        This method processes data, constructs appropriate folder paths based on
        provided parameters, and saves the data in the specified file format. The
        method enforces naming conventions for the file. It supports saving in CSV or JSON formats
        for different use cases.

        :param result_data: Input data to be saved.
        :param model_key: A unique string identifier for the model, specified in the path_config.
        :param parameters: A dictionary containing parameters used for constructing
                           folder hierarchy and metadata about the file.
        :param file_type: Specifies the type of file to save. Valid options are
                          "dataframe", "dict", and "plot". Dataframes are saved as CSV,
                          dictionaries are saved as JSON, and plots are saved as PNG.
        :param file_prefix: An optional prefix, added to the filename for further
                            customization. Should contain enough information to infer
                            the source of the result (e.g., the filename of the split set)
                            Defaults to an empty string.
        :param file_suffix: An optional suffix, added to the filename. Should contain information
                            about the result type (e.g. "metrics" or "analysis_plot").
        :return: None
        """
        # Construct the path to the folder, where the file will be saved
        folder_path = self.pm.get_complex_ml_path(parameters, ["results", model_key], False, True)

        # Save the file depending on the filetype
        self.save_file_depending_on_filetype(file_type, folder_path, file_prefix, file_suffix, result_data)

    def save_metadata_analysis(self, result_data, model_key: str, parameters: dict,
                       file_type: str, file_prefix: str = "", file_suffix: str = "", outlier_run_name: str = None):
        """
        Save metadata analysis data to a specified file format and location.

        This method processes data, constructs appropriate folder paths based on
        provided parameters, and saves the data in the specified file format. The
        method enforces naming conventions for the file. It supports saving in CSV or JSON formats
        for different use cases. File name will be constructed as <prefix>_<suffix>.<extension>

        :param result_data: Input data to be saved.
        :param model_key: A unique string identifier for the model, used for constructing
                          folder paths.
        :param parameters: A dictionary containing parameters used for constructing
                           folder hierarchy and metadata about the file.
        :param file_type: Specifies the type of file to save. Valid options are
                          "dataframe", "dict", and "plot". Dataframes are saved as CSV,
                          dictionaries are saved as JSON, and plots are saved as PNG.
        :param file_prefix: An optional prefix added to the filename for further
                            customization. Should contain enough information to infer
                            the source of the result (e.g. the filename of the split set)
                            Defaults to an empty string.
        :param file_suffix: An optional suffix added to the filename. Should contain information
                            about the result type (e.g. "metrics" or "analysis_plot").
        :param outlier_run_name: The run name of the metadata analysis run.
        :return: None
        """
        # Construct the path to the folder, where the file will be saved
        folder_path = self.pm.get_complex_ml_path(
            parameters, ["metadata_analysis", model_key], False, True, outlier_run_name
        )

        # Save the file depending on the filetype
        self.save_file_depending_on_filetype(file_type, folder_path, file_prefix, file_suffix, result_data)

    def save_predicted_set(self, test_df: pd.DataFrame, test_path: Path, pred_df, parameters: dict, model_key: str):
        """
        Save the predicted dataset along with necessary modifications and persist the results.

        This function appends predictions to the given test dataset, calculates the prediction
        error for each instance, and saves the modified dataset with additional metadata. It helps
        in evaluating model performance and preserving results for subsequent analysis.

        :param test_df: Pandas DataFrame representing the test dataset. Expected to have at least
                       columns including 'label' to compare with predictions.
        :param test_path: The file path of the original test dataset. Used to derive the filename for saving results.
        :param pred_df: Pandas Series or DataFrame representing the predicted labels generated
                       by the machine learning model.
        :param parameters: Parameters for the machine learning run. Saved along with results for traceability.
        :param model_key: Unique identifier of the model, which calculated the results.
        :return: None
        """
        test_df_copy = test_df.copy()
        test_df_copy["prediction"] = pred_df  # Append predicted labels to the test set
        # Add a prediction error column (1 = false pred, 2 = correct pred)
        test_df_copy["error"] = (test_df_copy["label"] != test_df_copy["prediction"]).astype(int)

        test_filename = test_path.stem
        self.save_ml_result(test_df_copy, model_key, parameters, "dataframe", test_filename, "full_and_pred")

    def save_run_metadata_to_json(self, parameters: dict, model_key: str, run_metadata: dict, filename: Path):
        """
        Save run metadata as a JSON file in the specified location.

        This method generates a file path by combining the provided folder structure with the given filename.
        The metadata is saved as a JSON file in the constructed path.

        :param parameters: Dict containing parameters used to generate the path.
        :param model_key: A string representing the specific model key associated with this run.
        :param run_metadata: The metadata of the run that needs to be saved.
        :param filename: The name of the file where the metadata will be saved.
        :return: None
        """
        folderpath = self.pm.get_complex_ml_path(parameters, ["run_metadata", model_key], False, True)
        fullpath = folderpath / filename
        self._save_data_as_json(run_metadata, fullpath)

    def save_global_outliers(self, parameters: dict, outliers_df: pd.DataFrame, outlier_type: str):
        """
        Saves the outliers over multiple runs. The file is saved in a specific path determined
        by the parameters and the type of outliers. Depending on the outlier type, the corresponding
        filename will be constructed. If the file already exists, the new data will be appended to the
        existing ones without duplication.

        :param parameters: Configuration parameters used to determine the file path and other settings
                         for saving the outlier data
        :param outliers_df: A DataFrame containing the outliers data that needs to be saved
        :param outlier_type: Type of outliers corresponding to its scope (I.e., outliers can be epochs or patients with
                            a critical number of outlier epochs). It must be either "epoch" or "patient_id".
                            Determines the filename for the saved data.
        :return: None
        """
        # Create the folder path
        folder_path = self.pm.get_complex_ml_path(parameters, ["global_outliers"], False, True)

        # Build a fullpath depending on the outlier type
        if outlier_type == "epoch":
            filename = "global_epoch_outliers.csv"
        elif outlier_type == "patient_id":
            filename = "global_patient_outliers.csv"
        else:
            raise ValueError("Invalid outlier type. Expected 'epoch' or 'patient_id'.")

        fullpath = Path(folder_path, filename)
        # Save outliers or append to an already existing file
        new_rows, new_rows_number = FileDataUtils.append_unique_rows_to_csv(outliers_df, fullpath)

        print(f"Saved outliers to {fullpath} with {new_rows_number} new rows")

    def save_further_analysis(self, hyperparameters: dict, results, result_type: str, analysis_key: str,
                              file_prefix: str, file_suffix: str):
        """
        Saves additional analysis data to the appropriate file and path based on the input parameters.

        :param hyperparameters: Dictionary of parameters that define the configuration for saving.
        :param results: The analysis data to be saved.
        :param result_type: File type of the result. Valid options are "dataframe", "dict", and "plot".
        :param analysis_key: A string key that determines the analysis that produced given results.
                            Valid options are "pca", "psd" and "k_means"
        :param file_prefix: Prefix for the file name to be saved
        :param file_suffix: Suffix for the file name to be saved
        :return: None
        """
        folder_path = self.pm.get_complex_ml_path(hyperparameters, ["further_analysis", analysis_key], False, True)

        self.save_file_depending_on_filetype(result_type, folder_path, file_prefix, file_suffix, results)

    def save_file_depending_on_filetype(self, file_type, folder_path, file_prefix, file_suffix, result_data):
        """
        Saves a file in a specified format based on the provided file type. This function supports saving
        dataframes, dictionaries, and plots. It generates the file name using the provided prefix and
        suffix, and the file is saved in the specified folder.

        :param file_type: Type of the file to be saved. Valid options are "dataframe", "dict", and "plot".
        :param folder_path: Path of the folder where the file should be saved.
        :param file_prefix: Prefix to be used in the generated file name.
        :param file_suffix: Suffix to be used in the generated file name.
        :param result_data: Data to be saved in the file. The format of this data must align with the
                             file type specified.
        :return: None
        :raises ValueError: If the provided file type is not supported.
        """
        # Choose the saving function based on the provided file_type
        if file_type == "dataframe":
            file_name = f"{file_prefix}_{file_suffix}.csv"
            saving_func = self._save_file_as_csv

        elif file_type == "dict":
            file_name = f"{file_prefix}_{file_suffix}.json"
            saving_func = self._save_data_as_json

        elif file_type == "plot":
            file_name = f"{file_prefix}_{file_suffix}.png"
            saving_func = self._save_plot

        else:
            raise ValueError(f"Unknown file type: {file_type}. Valid options are: dataframe, dict, plot")

        # Assemble the path and save the file
        fullpath = Path(folder_path, file_name)
        saving_func(result_data, fullpath)
        print(f"Successfully saved {file_type} to {fullpath}")

    @staticmethod
    def _save_file_as_csv(data, fullpath, index=True):
        """Saves data as a csv file in given fullpath."""
        data.to_csv(fullpath, index=index)

    @staticmethod
    def _save_plot(fig, fullpath: str):
        """
        Saves a matplotlib Figure to file.

        :param fig: The matplotlib Figure object to save.
        :param fullpath: Full path including filename and extension (e.g. 'figures/plot.png').
        """
        if fig is not None:
            fig.savefig(fullpath, dpi=300)
            plt.close(fig)
        else:
            print(f"Nothing to save at {fullpath}.")

    @staticmethod
    def _save_data_as_json(data, fullpath):
        serial_result_data = FileDataUtils.serialize_for_json(data)
        with open(fullpath, "w") as f:
            json.dump(serial_result_data, f, indent=4)