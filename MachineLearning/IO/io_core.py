from MachineLearning.Utils.path_utils import PathUtils
from MachineLearning.Utils.config_loader import load_config


class IOCore:
    path_config = load_config("path_config.yaml")
    data_names = load_config("data_names_config.yaml")

    def __init__(self, **kwargs):
        pass

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

    def psd_path_with_parameters(self, parameters: dict) -> str:
        """
        Returns a path to the PSD directory, with the current parameters specified in <parameters>.
        :param parameters: Current Episode Parameters
        :return: Path to the PSD directory
        """
        psd_dir = self.level2_subdir_path("features", "psds")
        abcd_subdir = PathUtils.create_A_B_C_D_subfolder_name("PSD", parameters)
        xy_subdir = PathUtils.create_X_Y_subfolder_name(parameters)
        return PathUtils.create_anypath(psd_dir, abcd_subdir, xy_subdir)

    def return_feature_name(self, feature_key: str) -> str:
        """Returns the name of the feature for given key from path_config"""
        return self.path_config["base_dir"]["subdirs"]["features"]["subdirs"][feature_key]

    def set_attributes(self, **kwargs):
        for attr in ["data_dir", "faw_subdir", "initial_data_subdir", "features_subdir", "plots_subdir",
                     "filtered_data_subdir", "psds_subdir"]:
            setattr(self, attr, kwargs.get(attr, getattr(self.__class__, attr)))
