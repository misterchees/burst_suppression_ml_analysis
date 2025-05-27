from MachineLearning.Utils.path_utils import PathUtils
from MachineLearning.Utils.config_loader import load_config


class IOCore:
    path_config = load_config("path_config.yaml")

    # field/column names for EEG files
    eeg_fs = "fs"
    eeg_rawEEG = "rawEEG"
    eeg_channels = ["1", "2"]

    # column names for PSDs
    psd_freq_col = "Frequency_Hz"
    psd_power_col = "PSD_V2_per_Hz"

    def __init__(self, **kwargs):
        pass

    def return_level1_subdir_path(self, subdir_key: str) -> str:
        """
        Returns a path to any direct subdirectory (defined by <subdir>) from the base directory of all data
        :param subdir_key: key that stores the name of the subdirectory. See path_config for more information.
        :return: path to the subdirectory
        """
        # retrieve subdir keys
        valid_subdirs = self.path_config["base_dir"]["subdirs"].keys()
        if subdir_key not in valid_subdirs:
            raise ValueError(f"Invalid subdir key: {subdir_key}. Valid subdir keys are {valid_subdirs}")
        base_dir = self.path_config["base_dir"]["path_name"]
        subdir = self.path_config["base_dir"]["subdirs"][subdir_key]
        return PathUtils.create_anypath(base_dir, subdir)

    def create_psd_path(self) -> str:
        return PathUtils.create_anypath(self.data_dir, self.features_subdir, self.psds_subdir)

    def create_psd_path_with_parameters(self, parameters: dict) -> str:
        psd_dir = self.create_psd_path()
        abcd_subdir = PathUtils.create_A_B_C_D_subfolder_name("PSD", parameters)
        xy_subdir = PathUtils.create_X_Y_subfolder_name(parameters)
        return PathUtils.create_anypath(psd_dir, abcd_subdir, xy_subdir)

    def set_attributes(self, **kwargs):
        for attr in ["data_dir", "faw_subdir", "initial_data_subdir", "features_subdir", "plots_subdir",
                     "filtered_data_subdir", "psds_subdir"]:
            setattr(self, attr, kwargs.get(attr, getattr(self.__class__, attr)))
