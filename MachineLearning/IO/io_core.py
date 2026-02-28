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
        self.path_manager = PathManager()

    def return_path_info(self, folder_keys: List[str], stem = False) -> Path | str:
        """
        Return information of the path for given keys in folder structure.

        :param folder_keys: List of keys to define the path.
        :param stem: If True, only the stem of the path is returned. Defaults to False.
        :return: Path as a Path or stem as a string.
        """
        path = self.path_manager.get_path(*folder_keys)
        if stem:
            path = str(path.stem)
        return path

    def return_all_feature_keys(self) -> list:
        """Returns a list of all features keys from the path config."""
        return list(self.path_config["root"]["features"]["children"].keys())

    def return_file_fullpath(self, parameters: dict, last_node_file: bool, create_subdirs: bool,
                             file_type: str, folder_keys: list[str]) -> Path:
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
            fullpath = self.return_all_parameter_fullpath(parameters, last_node_file, create_subdirs, folder_keys)
        elif file_type == "awake" or file_type == "normal_an":
            fullpath = self.return_no_parameters_fullpath(parameters, file_type, last_node_file, folder_keys)
        else:
            raise ValueError(f"file_type {file_type} not recognized. Valid options: 'faw', 'awake', 'normal_an'")

        return fullpath

    def return_no_parameters_fullpath(self, parameters: dict, file_type: str,
                                      last_node_file=True, folder_keys: List[str]=None) -> Path:
        """
       Creates a filepath for a file depending on parameters, file type, and folder keys,
       where the keys determine the folder and the parameters determine the name of the file.
       :param parameters: Parameters of the file
       :param file_type: Type of the file. Valid values are: 'awake' and 'normal_an'
       :param last_node_file: Is the last node a file or a folder. If True File, else Folder
       :param folder_keys: Keys that determine the folder. Every key is one level deeper in the directory structure
       :return: The assembled filepath as a Path object
       """
        # Get the correct filename based on filetype and parameters
        last_node = self._return_node_name(parameters, file_type)

        # Get the correct folder path based on folder_keys
        folder_dir = self.return_path_info(folder_keys)
        last_node = f"{last_node}.csv" if last_node_file else last_node
        output_path = Path(folder_dir, last_node)
        return output_path

    def return_all_parameter_fullpath(self, parameters: dict, last_node_file: bool, create_dirs: bool,
                                      folder_parts: List[str], run_name: str = None) -> Path:
        """
        Returns a fullpath that is of the following structure:
        folder1/folder2/...folderN/<folderN name_A_B_C_D/<episode Name>_X_Y
        :param parameters: Dictionary of parameters that defines A, B, C, D, X, Y
        :param last_node_file: Flag to determine if the last node is a csv file or a directory.
         -> fullpath ends then with .../<episode Name>_X_Y.csv
        :param create_dirs: Flag to create the necessary folders if not already present.
        :param folder_parts: Keys of folders from the path_config that define folder1 to folderN
        :param run_name: Name of the run. If None, will be loaded from parameters_config.yaml.
        :return: Return a fullpath of the above description.
        """
        # Keys of stages that are stored in individual runs (i.e., the corresponding run folders)
        individual_run_keys = ["splits", "models", "results", "metadata_analysis"]
        # Prepare all individual parts of the path to assemble
        dir_first_part = self.return_path_info(folder_parts)
        prefix_name = dir_first_part.stem
        dir_abcd_part = PathUtils.return_A_B_C_D_path(prefix_name, parameters)
        xy_part = PathUtils.return_X_Y_name(parameters)

        # X_Y (Parameters of individual epochs) can be either a directory or a file
        if not last_node_file:
            folder_path = Path(dir_first_part, dir_abcd_part, xy_part)
            fullpath = folder_path
        else:
            folder_path = Path(dir_first_part, dir_abcd_part)
            fullpath = Path(folder_path, f"{xy_part}.csv")

        # Append run_name if given. If not, use the current run name if it's from a specific filepath
        if run_name is not None:
            fullpath = Path(fullpath, run_name)
        elif any(key in folder_parts for key in individual_run_keys):
            run_name = load_config("parameters_config.yaml")["run_name"]
            if run_name is None:
                raise ValueError("Expected run_name in parameters_config but got None.")
            fullpath = Path(fullpath, run_name)

        if create_dirs:
            Path.mkdir(fullpath.parent if last_node_file else fullpath, exist_ok=True)

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

        full_folder_path = self.return_all_parameter_fullpath(parameters, False, create_subdirs,
                                                              ["test_and_train_data", "splits"])

        if train_or_test != "train" and train_or_test != "test":
            raise ValueError(f"train_or_test must be either 'train' or 'test'")
        return Path(full_folder_path, f"{train_or_test}_split.csv")

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

        full_folder_path = self.return_all_parameter_fullpath(parameters, False, create_subdirs,
                                                              ["test_and_train_data", "splits"])

        if train_or_test != "train" and train_or_test != "test":
            raise ValueError(f"train_or_test must be either 'train' or 'test'")
        return Path(full_folder_path, f"{fold_idx}_{total_folds}_{train_or_test}_split.csv")

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
        files_folderpath = self.return_all_parameter_fullpath(
            hyperparameters, False, False, folder_parts, run_name
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

    def return_run_metadata_fullpath(self, hyperparameters: dict, run_name: str, model_key: str) -> Path:
        """
        Returns the full path to the metadata file of a specific run based on the provided
        hyperparameters, run name, and model key.

        :param hyperparameters: A dictionary containing the hyperparameters of the run.
        :param run_name: The name of the specific run for which metadata is being retrieved.
        :param model_key: The key identifying the model.
        :return: The full file path to the metadata for the specified run.
        :raises FileNotFoundError: If no metadata file matching the given run name is found.
        """
        # Get all paths
        metadata_folderpath = self.return_all_parameter_fullpath(
            hyperparameters, False, False, ["run_metadata", model_key]
        )
        metadata_fullpaths, _ = PathUtils.list_files_in_folder(metadata_folderpath, ".json", fullpaths=True)

        # Search for a specific run metadata path
        matching_metadata_path = next(
            (path for path in metadata_fullpaths if path.name == f"{run_name}.json"),
            None
        )
        # Error if not found
        if not matching_metadata_path:
            raise FileNotFoundError(
                f"No matching file for run_name='{run_name}' found in folder='{metadata_folderpath}'.")

        return matching_metadata_path

    def return_all_patient_ids(self, initial_data_key: str) -> List[int]:
        """
        Returns a list of all patient IDs found in a directory specified by initial_data_key.
        :param initial_data_key: Key of the folder in initial data, which determines the path to the directory.
        :return: List of all patient IDs in the directory.
        """

        patient_ids = []
        directory = self.return_path_info(["initial_data", initial_data_key])

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
        if epoch_type != "faw":
            folder_path = self.return_no_parameters_fullpath(
                parameters, epoch_type, False, ["features", "psds"]
            )
        elif epoch_type == "faw":
            folder_path = self.return_all_parameter_fullpath(
                parameters, False, False, ["features", "psds"]
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