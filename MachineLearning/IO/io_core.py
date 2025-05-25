from MachineLearning.Utils.path_utils import PathUtils


class IOCore:
    # basic directory for all data in this project
    data_dir = ("C:\\Users\\jesus\\OneDrive\\Dokumente\\Jesús\\Studium\\Fächer - Bioinformatik\\"
                "Praktische Arbeit und Bachelorarbeit\\Material\\Daten")
    # subdirectory of subsets with episodes of fake awakeness (FAW)
    faw_subdir = "FAW_subsets"
    # subdirectory for all initial raw data only
    initial_data_subdir = "Initial_data"
    # subdirectory for features and related data (like PSDs for freq domain features)
    features_subdir = "Features"
    # subdirectory for plots
    plots_subdir = "Plots"
    # subdirectory for filtered data
    filtered_data_subdir = "Filtered"
    # subdirectory for PSDs
    psds_subdir = "PSDs"

    # field/column names for EEG files
    eeg_fs = "fs"
    eeg_rawEEG = "rawEEG"
    eeg_channels = ["1", "2"]

    # column names for PSDs
    psd_freq_col = "Frequency_Hz"
    psd_power_col = "PSD_V2_per_Hz"

    def __init__(self, **kwargs):
        """
        Initialize the IOCore with optional variables. It checks for every attribute
        and uses the initialized default if no new value is passed.
        """
        for attr in ["data_dir", "faw_subdir", "initial_data_subdir", "features_subdir", "plots_subdir",
                     "filtered_data_subdir", "psds_subdir"]:
            setattr(self, attr, kwargs.get(attr, getattr(self.__class__, attr)))

    def create_faw_path(self) -> str:
        return PathUtils.create_anypath(self.data_dir, self.faw_subdir)

    def create_initial_data_path(self) -> str:
        return PathUtils.create_anypath(self.data_dir, self.initial_data_subdir)

    def create_features_path(self) -> str:
        return PathUtils.create_anypath(self.data_dir, self.features_subdir)

    def create_plots_path(self) -> str:
        return PathUtils.create_anypath(self.data_dir, self.plots_subdir)

    def create_filtered_data_path(self) -> str:
        return PathUtils.create_anypath(self.data_dir, self.filtered_data_subdir)

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
