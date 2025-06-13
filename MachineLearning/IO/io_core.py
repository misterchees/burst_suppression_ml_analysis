import os

from MachineLearning.Utils.path_utils import PathUtils
from MachineLearning.Utils.config_loader import load_config


class IOCore:
    def __init__(self, **kwargs):
        self.path_config = load_config("path_config.yaml")
        self.data_names = load_config("data_names_config.yaml")

    def level1_subdir_path(self, subdir_key: str) -> str:
        """
        Returns a path to any direct subdirectory (defined by <subdir>) from the base directory of all data.
        See path_config for more information.
        :param subdir_key: key that stores the name of the subdirectory.
        :return: path to the subdirectory
        """
        # retrieve subdir keys
        valid_subdirs = self.path_config["base_dir"]["subdirs"].keys()

        if subdir_key not in valid_subdirs:
            raise ValueError(f"Invalid lvl1 subdir key: {subdir_key}. Valid subdir keys are {valid_subdirs}")

        base_dir = self.path_config["base_dir"]["path_name"]
        subdir = self.path_config["base_dir"]["subdirs"][subdir_key]["path_name"]
        return PathUtils.create_anypath(base_dir, subdir)

    def level2_subdir_path(self, subdir_lvl_1_key: str, subdir_lvl_2_key) -> str:
        """
        Returns a path to any subdirectory two nodes away from the base directory of all data.
        See path_config for more information.
        :param subdir_lvl_1_key: key that stores the name of the direct subdirectory ot the base directory.
        :param subdir_lvl_2_key: key that stores the name of the direct subdirectory ot the lvl1 subdirectory.
        :return: path to the subdirectory
        """
        lvl_1_subdir = self.level1_subdir_path(subdir_lvl_1_key)
        # retrieve subdir keys
        valid_lvl2_subdirs = self.path_config["base_dir"]["subdirs"][subdir_lvl_1_key]["subdirs"].keys()

        if subdir_lvl_2_key not in valid_lvl2_subdirs:
            raise ValueError(f"Invalid lvl2 subdir key: {subdir_lvl_2_key}. Valid subdir keys are {valid_lvl2_subdirs}")

        lvl_2_subdir = self.path_config["base_dir"]["subdirs"][subdir_lvl_1_key]["subdirs"][subdir_lvl_2_key]
        return PathUtils.create_anypath(lvl_1_subdir, lvl_2_subdir)

    def psd_folder_path(self, parameters: dict, faw=True) -> str:
        """
        Returns a path to the PSD directory, with the current parameters specified in <parameters>.
        :param parameters: Current Episode Parameters
        :param faw: Boolean to determine if PSD is from fake awake or true awake
        :return: Path to the PSD directory
        """
        if faw:
            psd_dir = self.level2_subdir_path("features", "psds")
            abcd_subdir = PathUtils.create_A_B_C_D_name("PSD", parameters)
            xy_subdir = PathUtils.create_X_Y_name(parameters)
            output_path = PathUtils.create_anypath(psd_dir, abcd_subdir, xy_subdir)
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

        fullpath = PathUtils.create_anypath(root["path_name"], *nodes)
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
        folder_name = self.return_folder_name(subdir_lvl1_key, subdir_lvl2_key)
        folder_path = self.return_folder_path(subdir_lvl1_key, subdir_lvl2_key)
        abcd_subdir = PathUtils.create_A_B_C_D_name(folder_name, parameters)
        subdir_of_file = PathUtils.create_anypath(folder_path, abcd_subdir)
        fullpath = PathUtils.return_csv_fullpath(folder_path, folder_name, parameters)

        # create subfolder if it doesn't exist
        if create_subdirs:
            os.makedirs(subdir_of_file, exist_ok=True)

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
        psd_subfolder = PathUtils.create_awake_file_name(parameters)
        output_path = PathUtils.create_anypath(folder_dir, f"{psd_subfolder}.csv")

        return output_path

    def set_attributes(self, **kwargs):
        for attr in ["data_dir", "faw_subdir", "initial_data_subdir", "features_subdir", "plots_subdir",
                     "filtered_data_subdir", "psds_subdir"]:
            setattr(self, attr, kwargs.get(attr, getattr(self.__class__, attr)))
