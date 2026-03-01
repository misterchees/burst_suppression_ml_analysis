"""This module contains the IOCore class"""
from pathlib import Path
import warnings
from typing import List

from MachineLearning.Utils.path_utils import PathUtils
from MachineLearning.Utils.config_handler import load_config
from MachineLearning.Utils.path_manager import PathManager


class IOCore:
    """
    This class is a superclass for all classes that mainly manage IO functionality, especially
    saving and loading data.
    """

    def __init__(self):
        """Initializes the IOCore class with the path and data config to handle paths."""
        self.path_config = load_config("path_config.yaml")
        self.data_names = load_config("data_names_config.yaml")
        self.pm = PathManager()

    def return_file_fullpath(self, parameters: dict, last_node_file: bool, create_subdirs: bool,
                             file_type: str, folder_keys: List[str]) -> Path:
        """
        Returns a filepath depending on given parameters.
        :param parameters: Defines last subfolders and the filename containing metadata.
        :param last_node_file: If True, the last node of the path is a (csv)file, else a folder.
        :param create_subdirs: If true, it will create all subdirectories necessary of the returned path.
        :param file_type: Defines together with parameters the last subfolders and file. Valid options are:
        'normal_an', 'faw' and 'awake'
        :param folder_keys: Keys that define the first part of the path from the base directory.
        :return: The filepath as a Path object
        """
        if file_type == "faw":
            fullpath = self.pm.get_complex_ml_path(parameters, folder_keys, last_node_file, create_subdirs)
        elif file_type == "awake" or file_type == "normal_an":
            fullpath = self.pm.get_simple_episode_path(parameters, file_type, folder_keys, last_node_file)
        else:
            raise ValueError(f"file_type {file_type} not recognized. Valid options: 'faw', 'awake', 'normal_an'")

        return fullpath

    def return_single_split_folder_fullpath(self, parameters: dict, train_or_test: str, create_subdirs=True) -> Path:
        """
        Returns a fullpath to the current train or test file in the split folder (single split).
        Ensures consistent names of the split files
        :param parameters: Parameters that determine the subfolder path in the split folder.
        :param train_or_test: Defines if the path leads to a train or test file. Valid options are 'train' and 'test'
        :param create_subdirs: Flag to create the necessary folders if not already present.
        :return: The defined fullpath.
        """

        full_folder_path = self.pm.get_complex_ml_path(
            parameters,["test_and_train_data", "splits"], False, create_subdirs
        )

        if train_or_test != "train" and train_or_test != "test":
            raise ValueError(f"train_or_test must be either 'train' or 'test'")
        return full_folder_path / f"{train_or_test}_split.csv"

    def return_folded_split_folder_fullpath(self, parameters: dict, train_or_test: str,
                                            fold_idx: int, total_folds: int, create_subdirs=True) -> Path:
        """
        Returns a fullpath to the current train or test file in the split folder (part of a folded split).
        Ensures consistent names of the split files
        :param parameters: Parameters that determine the subfolder path in the split folder.
        :param train_or_test: Defines if the path leads to a train or test file. Valid options are 'train' and 'test'
        :param fold_idx: Current fold index
        :param total_folds: Sum of folds.
        :param create_subdirs: Flag to create the necessary folders if not already present.
        :return: The defined fullpath.
        """

        full_folder_path = self.pm.get_complex_ml_path(
            parameters, ["test_and_train_data", "splits"], False, create_subdirs
        )

        if train_or_test != "train" and train_or_test != "test":
            raise ValueError(f"train_or_test must be either 'train' or 'test'")
        return full_folder_path / f"{fold_idx}_{total_folds}_{train_or_test}_split.csv"

    def return_related_fullpaths(self, hyperparameters: dict, run_name: str, folder_parts: list) -> List[Path]:
        """
        Returns a list of file paths for related CSV files, such as train and test
        splits, located in the specified folder for a given run name.

        :param hyperparameters: A dictionary containing hyperparameters, used to construct the full path to the folder.
        :param run_name: The name of the current run to identify the folder containing the relevant files.
        :param folder_parts: A list of keys that define the path to the folder.
        :return: A list of file paths.
        :raises FileNotFoundError: If no valid files are found in the specified folder.
        """
        # Get folder of related files
        files_folderpath = self.pm.get_complex_ml_path(
            hyperparameters, folder_parts, False, False, run_name
        )
        fullpaths_list, _ = PathUtils.list_files_in_folder(files_folderpath, ".csv", fullpaths=True)

        # Filter out all files that are not relevant based on the folder they are in
        if folder_parts[-1] == "splits":
            relevant_paths = [
                path for path in fullpaths_list
                if path.name == "train_split.csv" or path.name == "test_split.csv"
            ]
        elif folder_parts[0] == "results":
            relevant_paths = [
                path for path in fullpaths_list
                if path.name == "full_and_pred.csv"
            ]
        else:
            raise ValueError(f"Unexpected folder to retrieve related files from: {files_folderpath}")
        # Validation if relevant files were found
        if not relevant_paths:
            raise FileNotFoundError(f"No valid files found in folder {files_folderpath} for run_name='{run_name}'")

        return relevant_paths

    def return_all_patient_ids(self, folder_keys: List[str]) -> List[int]:
        """
        Returns a list of all patient IDs found in a directory.
        Expects the directory to contain files named <patient_id>.<file_extension>.
        :param folder_keys: Keys of the directory with the patientID files.
        :return: List of all patient IDs in the directory.
        """

        patient_ids = []
        directory = self.pm.get_path(*folder_keys)

        for file in directory.iterdir():
            try:
                # Get Patient ID from the filename
                patient_id = int(file.name.split(".")[0])
                # Add filename to the patient ID list
                patient_ids.append(patient_id)
            except TypeError as te:
                warnings.warn(f"File: {file} has not the right format. It should be <integer>.<file extension>\n"
                              f"Error message: {te}")
            except Exception as ex:
                warnings.warn(f"Something went wrong while file: {file} was parsed. Error {ex}")

        return patient_ids

    def clear_psd_folder(self, parameters: dict, epoch_type: str):
        """
        Deletes all files (not folders) in the specified psd folder.

        :param parameters: Parameters that determine the subfolder path in the folder.
        :param epoch_type: Defines the PSD Folder that will be cleared. Valid options are 'awake', 'normal_an' and 'faw'
        """
        if epoch_type == "awake" or epoch_type == "normal_an":
            folder_path = self.pm.get_simple_episode_path(
                parameters, epoch_type, ["features", "psds"], False
            )
        elif epoch_type == "faw":
            folder_path = self.pm.get_complex_ml_path(
                parameters, ["features", "psds"], False, False
            )
        else:
            raise ValueError(f"Unknown epoch type: {epoch_type}. "
                             "Epoch_type must be either 'awake', 'normal_an' or 'faw'")

        PathUtils.clear_folder(folder_path)

    @staticmethod
    def _return_node_name(parameters: dict, node_type: str) -> str:
        """
        Return a formatted node name based on the fixed_window_size field of given parameters and node type.

        :param parameters: A dictionary containing the configuration parameters.
        :param node_type: Specifies the type of node. Valid values are 'awake' and 'normal_an'.

        :return: A formatted string representing the node name based on the node type and
            epoch length.

        :raises ValueError: If the provided node type is not recognized.
        """
        epoch_length = parameters["fixed_window_size"]
        if node_type == "awake":
            return f"Awake_{epoch_length}"
        elif node_type == "normal_an":
            return f"Normal_ane_{epoch_length}"
        raise ValueError(f"Unknown node type {node_type}. Valid types are 'awake' and 'normal_an'")