"""This module contains the IOCore class"""
import os
import warnings
from MachineLearning.Utils.path_utils import PathUtils
from MachineLearning.Utils.config_handler import load_config


class IOCore:
    """
    This class is a superclass for all classes that mainly manage IO stuff. It provides functions that are
    useful for saving and loading data, which is mostly path manipulation methods.
    """

    def __init__(self):
        """Intializes the IOCore class with the path and data config to handle paths."""
        self.path_config = load_config("path_config.yaml")
        self.data_names = load_config("data_names_config.yaml")

    def psd_folder_path(self, parameters: dict, epoch_type: str) -> str:
        """
        Returns a path to the PSD directory, with the current parameters specified in <parameters>.
        :param parameters: Current Episode Parameters
        :param epoch_type: Type of Episode, that influences the path.
        :return: Path to the PSD directory
        """
        output_path = self.return_file_fullpath(parameters, False, True, epoch_type, ["features", "psds"])
        return output_path

    def return_feature_name(self, feature_key: str) -> str:
        """Returns the name of the feature for given key from path_config"""
        return self.return_folder_name("features", feature_key)

    def return_folder_name(self, *folder_keys: str) -> str:
        """
        Returns the name of the folder for given folder_keys from path_config by recursively traveling
        one node deeper for every key.
        :param folder_keys: Any number of folder keys.
        :return: Name of the folder
        """
        # Check if keys given
        if len(folder_keys) == 0:
            raise ValueError(f"No folder keys specified.")

        current_node = self.path_config["base_dir"]
        for key in folder_keys:
            current_node = current_node["subdirs"][key]

        # If it's a leaf in the yaml config, then its key:value, else it's key:dict with name in key = path_name
        if isinstance(current_node, dict):
            return current_node["path_name"]
        else:
            return current_node

    def return_folder_path(self, folder_keys: list[str]) -> str:
        """
        Returns a fullpath to the last folder given in folder keys, following all folders along the way.
        Returns the base dir if no arguments are given.
        :param folder_keys: Keys to assemble the path of folders
        :return: Fullpath as string
        """
        root = self.path_config["base_dir"]
        current_node = root
        nodes = []
        for key in folder_keys:
            current_node = current_node["subdirs"][key]
            # If it's a leaf in the yaml config, then its key:value, else it's key:dict with name in key = path_name
            if isinstance(current_node, dict):
                nodes.append(current_node["path_name"])
            else:
                nodes.append(current_node)

        fullpath = PathUtils.return_anypath(root["path_name"], *nodes)
        return fullpath

    def return_all_feature_keys(self) -> list:
        """Returns a list of all features keys from the path config."""
        return list(self.path_config["base_dir"]["subdirs"]["features"]["subdirs"].keys())

    def return_file_fullpath(self, parameters: dict, last_node_file: bool, create_subdirs: bool,
                             file_type: str, folder_keys: list[str]) -> str:
        """
        Returns a filepath depending on given parameters.
        :param parameters: Defines last subfolders and the filename containing metadata.
        :param last_node_file: If True, the last node of the path is a (csv)file, else a folder.
        :param create_subdirs: If true, will create all subdirectories necessary of returned path.
        :param file_type: Defines together with parameters the last subfolders and file. Valid options are:
        'normal_an', 'faw' and 'awake'
        :param folder_keys: Keys that define the first part of the path from the base directory.
        :return: The filepath as a String.
        """
        if file_type == "faw":
            fullpath = self.return_all_parameter_fullpath(parameters, last_node_file, create_subdirs, folder_keys)
        elif file_type == "awake" or file_type == "normal_an":
            fullpath = self.return_no_parameters_fullpath(parameters, file_type, last_node_file, folder_keys)
        else:
            raise ValueError(f"file_type {file_type} not recognized. Valid options: 'faw', 'awake', 'normal_an'")

        return fullpath

    def return_no_parameters_fullpath(self, parameters: dict, file_type: str,
                                      last_node_file=True, folder_keys: list[str]=None) -> str:
        """
       Creates a filepath for a file depending on parameters, file type and folder keys,
       where the keys determine the folder and the parameters determine the name of the file.
       :param parameters: parameters of the file
       :param file_type: Type of the file. Valid values are: 'awake' and 'normal_an'
       :param last_node_file: Is last node a file or a folder. If True File, else Folder
       :param folder_keys: keys that determine the folder. Every key is one level deeper in the directory structure
       :return: Path to the csv file
       """
        # Get subfolder. Only difference is the file name. Folder is the same.
        last_node = PathUtils.return_node_name(parameters, file_type)

        folder_dir = self.return_folder_path(folder_keys)
        last_node = f"{last_node}.csv" if last_node_file else last_node
        output_path = PathUtils.return_anypath(folder_dir, last_node)
        return output_path

    def return_all_parameter_fullpath(self, parameters: dict, last_node_file: bool, create_dirs: bool,
                                      folder_parts: list[str], run_name: str = None) -> str:
        """
        Returns a fullpath that is of following structure:
        folder1/folder2/...folderN/<folderN name_A_B_C_D/<episode Name>_X_Y
        :param parameters: Dictionary of parameters that defines A, B, C, D, X, Y
        :param last_node_file: Flag to determine if the last node is a csv file.
         -> fullpath ends then with .../<episode Name>_X_Y.csv
        :param create_dirs: Flag to create the necessary folders if not already present.
        :param folder_parts: Keys of folders from the path_config that define folder1 to folderN
        :param run_name: Name of the run. If None, will be loaded from parameters_config.yaml.
        :return: Return a fullpath of the above description.
        """
        individual_run_keys = ["splits", "models", "results", "metadata_analysis"]  # Keys to create individual run folders
        dir_first_part = self.return_folder_path(folder_parts)
        prefix_name = self.return_folder_name(*folder_parts)

        if not last_node_file:
            dir_abcd_xy_part = PathUtils.return_A_B_C_D_X_Y_path(prefix_name, parameters)
            folder_path = PathUtils.return_anypath(dir_first_part, dir_abcd_xy_part)
            fullpath = folder_path
        else:
            dir_abcd_part = PathUtils.return_A_B_C_D_name(prefix_name, parameters)
            folder_path = PathUtils.return_anypath(dir_first_part, dir_abcd_part)
            xy_part = PathUtils.return_X_Y_name(parameters)
            fullpath = PathUtils.return_anypath(folder_path, f"{xy_part}.csv")

        # Append run_name if given. If not, use the current run name if it's from a specific filepath
        if run_name is not None:
            fullpath = PathUtils.return_anypath(fullpath, run_name)
        elif any(key in folder_parts for key in individual_run_keys):
            run_name = load_config("parameters_config.yaml")["run_name"]
            if run_name is None:
                raise ValueError("Expected run_name in parameters_config but got None.")
            fullpath = PathUtils.return_anypath(fullpath, run_name)

        if create_dirs:
            os.makedirs(os.path.dirname(fullpath) if last_node_file else fullpath, exist_ok=True)

        return fullpath

    def return_single_split_folder_fullpath(self, parameters: dict, train_or_test: str, create_subdirs=True) -> str:
        """
        Returns a fullpath to current train or test file in the split folder.
        :param parameters: Parameters that determine the subfolder path in the split folder.
        :param train_or_test: Defines if path leads to a train or test file. Valid options are 'train' and 'test'
        :param create_subdirs: Flag to create the necessary folders if not already present.
        :return: The defined fullpath as string.
        """

        full_folder_path = self.return_all_parameter_fullpath(parameters, False, create_subdirs,
                                                              ["test_and_train_data", "splits"])

        if train_or_test != "train" and train_or_test != "test":
            raise ValueError(f"train_or_test must be either 'train' or 'test'")
        return PathUtils.return_anypath(full_folder_path, f"{train_or_test}_split.csv")

    def return_folded_split_folder_fullpath(self, parameters: dict, train_or_test: str,
                                            fold_idx: int, total_folds: int, create_subdirs=True) -> str:
        """
        Returns a fullpath to current train or test file in the split folder.
        :param parameters: Parameters that determine the subfolder path in the split folder.
        :param train_or_test: Defines if path leads to a train or test file. Valid options are 'train' and 'test'
        :param fold_idx: Current fold index
        :param total_folds: Sum of folds.
        :param create_subdirs: Flag to create the necessary folders if not already present.
        :return: The defined fullpath as string.
        """

        full_folder_path = self.return_all_parameter_fullpath(parameters, False, create_subdirs,
                                                              ["test_and_train_data", "splits"])

        if train_or_test != "train" and train_or_test != "test":
            raise ValueError(f"train_or_test must be either 'train' or 'test'")
        return PathUtils.return_anypath(full_folder_path, f"{fold_idx}_{total_folds}_{train_or_test}_split.csv")

    def return_related_fullpaths(self, hyperparameters: dict, run_name: str, folder_parts: list) -> list:
        """
        Returns a list of file paths for related CSV files, such as train and test
        splits, located in the specified folder for a given run name.

        :param hyperparameters: A dictionary containing various hyperparameters, which
                                will be used to construct the full path to the folder.
        :type hyperparameters: dict
        :param run_name: The name of the current run to identify the folder containing
                         the relevant files.
        :type run_name: str
        :param folder_parts: A list of keys that define the path to the folder.
        :type folder_parts: list
        :return: A list of file paths.
        :rtype: list
        :raises FileNotFoundError: If no valid files are found in the specified folder.
        """
        from pathlib import Path
        # Get folder of related files
        files_folderpath = self.return_all_parameter_fullpath(
            hyperparameters, False, False, folder_parts, run_name
        )
        fullpaths_list, _ = PathUtils.list_files_in_folder(files_folderpath, ".csv", fullpaths=True)

        # Filter out all files that are not relevant based on folder they are in
        if folder_parts[-1] == "splits":
            relevant_paths = [
                path for path in fullpaths_list
                if os.path.basename(path).endswith(("train_split.csv", "test_split.csv"))
            ]
        elif folder_parts[0] == "results":
            relevant_paths = [
                path for path in fullpaths_list
                if os.path.basename(path).endswith("full_and_pred.csv")
            ]
        else:
            raise ValueError(f"Unexpected folder to retrieve related files from: {files_folderpath}")
        # Validation for split folder
        if not relevant_paths:
            raise FileNotFoundError(f"No valid files found in folder {files_folderpath} for run_name='{run_name}'")

        # Convert Strings to Path Objects
        relevant_paths = [Path(p) for p in relevant_paths]

        return relevant_paths

    def return_run_metadata_fullpath(self, hyperparameters: dict, run_name: str, model_key: str) -> str:
        """
        Returns the full path to the metadata file of a specific run based on the provided
        hyperparameters, run name, and model key.

        :param hyperparameters: A dictionary containing the hyperparameters of the run.
        :param run_name: The name of the specific run for which metadata is being retrieved.
        :param model_key: The key identifying the model.
        :return: The full file path to the metadata for the specified run.
        :rtype: str
        :raises FileNotFoundError: If no metadata file matching the given run name is found.
        """
        # Get all paths
        metadata_folderpath = self.return_all_parameter_fullpath(
            hyperparameters, False, False, ["run_metadata", model_key]
        )
        metadata_fullpaths, _ = PathUtils.list_files_in_folder(metadata_folderpath, ".json", fullpaths=True)

        # Search for a specific run metadata path
        matching_metadata_path = next(
            (path for path in metadata_fullpaths if os.path.basename(path) == f"{run_name}.json"),
            None
        )
        # Error if not found
        if not matching_metadata_path:
            raise FileNotFoundError(
                f"No matching file for run_name='{run_name}' found in folder='{metadata_folderpath}'.")

        return matching_metadata_path

    def return_all_patient_ids(self, initial_data_key: str) -> list:
        """
        Returns a list of all patient IDs found in a directory specified by initial_data_key.
        :param initial_data_key: Key of the folder in initial data, that determines the path to the directory.
        :return: List of all patient IDs in directory.
        """

        patient_ids = []
        directory = self.return_folder_path(["initial_data", initial_data_key])

        for file in os.listdir(directory):
            try:
                # Try to get Patient ID from filename
                patient_id = int(file.split(".")[0])
                # Add filename to list
                patient_ids.append(patient_id)
            except TypeError as ex:
                warnings.warn(f"File: {file} has not the right format. It should be <integer>.<file extension>\n"
                              f"Error message: {ex}")
            except Exception as ex:
                warnings.warn(f"Something went wrong while file: {file} was parsed. Error {ex}")

        return patient_ids

    def clear_psd_folder(self, parameters: dict, epoch_type: str):
        """
        Deletes all files (not folders) in the specified psd folder, that contains normal anesthesia data.
        Only folders of this type will be cleared because, normal anesthesia data are created everytime
        completely random and in the worst case, new data can mix with old data.

        :param parameters: Parameters that determine the subfolder path in the folder.
        :param epoch_type: Defines the PSD Folder that will be cleared. Valid options are 'awake', 'normal_an' and 'faw'
        """
        if epoch_type != "faw":
            folder_path = self.return_no_parameters_fullpath(parameters, epoch_type, False, ["features", "psds"])
        elif epoch_type == "faw":
            folder_path = self.return_all_parameter_fullpath(parameters, False, False, ["features", "psds"])
        else:
            raise ValueError(f"Unknown epoch type: {epoch_type}. "
                             "Epoch_type must be either 'awake', 'normal_an' or 'faw'")

        PathUtils.clear_folder(folder_path)
