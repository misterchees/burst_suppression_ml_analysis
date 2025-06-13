import os

from MachineLearning.Utils.path_utils import PathUtils
from MachineLearning.Utils.config_loader import load_config


class IOCore:
    def __init__(self, **kwargs):
        self.path_config = load_config("path_config.yaml")
        self.data_names = load_config("data_names_config.yaml")

    def psd_folder_path(self, parameters: dict, faw=True) -> str:
        """
        Returns a path to the PSD directory, with the current parameters specified in <parameters>.
        :param parameters: Current Episode Parameters
        :param faw: Boolean to determine if PSD is from fake awake or true awake
        :return: Path to the PSD directory
        """
        if faw:
            output_path = self.return_all_parameter_fullpath(parameters, False, True,"features", "psds")
        else:
            output_path = self.return_awake_file_fullpath(parameters, "features", "psds")

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

    def return_folder_path(self, *folder_keys: str) -> str:
        """
        Returns a fullpath to the last folder given in folder keys, following all folders along the way.
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

    def return_all_feature_names(self) -> list:
        """Returns a list of all feature names available in path_config"""
        return list(self.path_config["base_dir"]["subdirs"]["features"]["subdirs"].values())

    def return_all_feature_keys(self) -> list:
        return list(self.path_config["base_dir"]["subdirs"]["features"]["subdirs"].keys())

    def return_faw_file_fullpath(self, parameters: dict, subdir_lvl1_key: str, subdir_lvl2_key: str,
                                 create_subdirs=True) -> str:
        """
        Returns a file path to a csv file for given foldername keys from path_config (2 nodes away from base directory)
        and given parameters. It creates all necessary folders if not already present.
        :param parameters: Parameters for the episodes
        :param subdir_lvl1_key: The key for the subdirectory of the base directory.
        :param subdir_lvl2_key: The key for the subdirectory of the lvl1 subdirectory.
        :param create_subdirs: If True will create all subdirectories necessary of returned path.
        :return: Path to the feature csv file
        """

        fullpath = self.return_all_parameter_fullpath(parameters, True, create_subdirs,
                                                      subdir_lvl1_key, subdir_lvl2_key)
        return fullpath

    def return_awake_file_fullpath(self, parameters: dict, *folder_keys) -> str:
        """
        Creates a filepath for an awake file depending on parameters and folder keys, where the keys determine
        the folder and the parameters determine the name of the file.
        :param parameters: parameters of the file
        :param folder_keys: keys that determine the folder. Every key is one level deeper in the directory structure
        :return:
        """
        folder_dir = self.return_folder_path(*folder_keys)
        psd_subfolder = PathUtils.return_awake_file_name(parameters)
        output_path = PathUtils.return_anypath(folder_dir, f"{psd_subfolder}.csv")

        return output_path

    def return_split_folder_fullpath(self, parameters: dict, train_or_test: str, create_subdirs=True) -> str:

        full_folder_path = self.return_all_parameter_fullpath(parameters, False, create_subdirs,
                                                              "test_and_train_data", "splits")

        if train_or_test != "train" and train_or_test != "test":
            raise ValueError(f"train_or_test must be either 'train' or 'test'")

        return PathUtils.return_anypath(full_folder_path, f"{train_or_test}_split.csv")

    def return_all_parameter_fullpath(self, parameters: dict, xy_file: bool, create_dirs: bool,
                                      *folder_parts: str) -> str:
        """
        Returns a fullpath that is of following structure:
        folder1/folder2/...folderN/<folderN name_A_B_C_D/<episode Name>_X_Y
        :param parameters: Dictionary of parameters that defines A,B,C,D,X,Y
        :param xy_file: Flag to determine, if last node is a csv. -> fullpath ends then with .../<episode Name>_X_Y.csv
        :param create_dirs: Flag to create the necessary folders if not already present.
        :param folder_parts: Keys of folders from path_config, that define folder1 to folderN
        :return: Returns a fullpath of above description.
        """
        dir_first_part = self.return_folder_path(*folder_parts)
        prefix_name = self.return_folder_name(*folder_parts)
        if not xy_file:
            dir_abcd_xy_part = PathUtils.return_A_B_C_D_X_Y_path(prefix_name, parameters)
            folder_path = PathUtils.return_anypath(dir_first_part, dir_abcd_xy_part)
            fullpath = folder_path
        else:
            dir_abcd_part = PathUtils.return_A_B_C_D_name(prefix_name, parameters)
            folder_path = PathUtils.return_anypath(dir_first_part, dir_abcd_part)
            xy_part = PathUtils.return_X_Y_name(parameters)
            fullpath = PathUtils.return_anypath(folder_path, f"{xy_part}.csv")

        if create_dirs:
            os.makedirs(folder_path, exist_ok=True)

        return fullpath

    def set_attributes(self, **kwargs):
        for attr in ["data_dir", "faw_subdir", "initial_data_subdir", "features_subdir", "plots_subdir",
                     "filtered_data_subdir", "psds_subdir"]:
            setattr(self, attr, kwargs.get(attr, getattr(self.__class__, attr)))
