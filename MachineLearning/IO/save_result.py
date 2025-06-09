import numpy as np
import os
import pandas as pd
from MachineLearning.IO.io_core import IOCore
from MachineLearning.Utils.path_utils import PathUtils


class SaveResult(IOCore):
    def __init__(self):
        super().__init__()

    def save_psd(self, frequencies: np.ndarray, power: np.ndarray,
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
        psd_dir = self.level2_subdir_path("features", "psds")
        abcd_subdir = PathUtils.create_A_B_C_D_subfolder_name("PSD", parameters)
        xy_subdir = PathUtils.create_X_Y_subfolder_name(parameters)
        psd_dir_fullpath = PathUtils.create_anypath(psd_dir, abcd_subdir, xy_subdir)

        # make sure directory exists
        os.makedirs(psd_dir_fullpath, exist_ok=True)

        # create dataframe from PSD data
        psd_cols = self.data_names["psd_files"]
        psd_df = pd.DataFrame({
            psd_cols["psd_freq_col"]: frequencies,
            psd_cols["psd_power_col"]: power
        })

        # create fullpath with PSD name to save data
        psd_filename = f"PSD_{start}_{end}_{result_id}.csv"
        fullpath = PathUtils.create_anypath(psd_dir_fullpath, psd_filename)

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
        psd_dir = self.level2_subdir_path("features", "psds")

        if filtered:
            filter_prefix = "filtered"
        else:
            filter_prefix = "raw"

        psd_filename = f"PSD_{filter_prefix}_whole_EEG_{result_id}.csv"
        whole_eeg_subdir = PathUtils.create_anypath(psd_dir, "whole_EEG_PSD", filter_prefix)

        # make sure directory exists
        os.makedirs(whole_eeg_subdir, exist_ok=True)

        # create dataframe from PSD data
        psd_cols = self.data_names["psd_files"]
        psd_df = pd.DataFrame({
            psd_cols["psd_freq_col"]: frequencies,
            psd_cols["psd_power_col"]: power
        })

        # create fullpath with PSD name to save data
        fullpath = PathUtils.create_anypath(whole_eeg_subdir, psd_filename)

        # save psd
        psd_df.to_csv(fullpath, index=False)

        print(f"Single episode PSD saved: {fullpath}")

    def save_feature_summary_episode(self, results: list, feature_key: str, parameters: dict):
        """
        Saves a csv with all episodes of a given parameter combination
        :param results: list of results
        :param feature_key: key of the subdirectory in path_config, where the summary episode will be saved
        :param parameters: parameters for the episode -> define the subfolder names
        """
        # save as CSV in subfolder with prefix
        result_df = pd.DataFrame(results)
        feature_name = self.return_feature_name(feature_key)

        subdir_of_feature = self.level2_subdir_path("features", feature_key)
        abcd_subdir = PathUtils.create_A_B_C_D_subfolder_name(feature_name, parameters)
        subdir_of_file = PathUtils.create_anypath(subdir_of_feature, abcd_subdir)
        fullpath = PathUtils.create_csv_fullpath(subdir_of_feature, feature_name, parameters)

        os.makedirs(subdir_of_file, exist_ok=True)
        result_df.to_csv(fullpath, index=False)  # write without row index

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
        filtered_eeg_subdir = self.level1_subdir_path("filtered_data")
        os.makedirs(os.path.dirname(filtered_eeg_subdir), exist_ok=True)
        fullpath = PathUtils.create_anypath(filtered_eeg_subdir, f"{result_id}.csv")

        # Write fs as header line, then the rounded data
        with open(fullpath, "w", newline='') as f:
            f.write(f"# fs = {fs}\n")
            df.to_csv(f, index=False, float_format="%.4f")

        print(f"Successfully saved filtered EEG to {fullpath}")
