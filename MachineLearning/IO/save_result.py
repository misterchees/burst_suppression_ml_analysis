import numpy as np
import os
import pandas as pd
from MachineLearning.IO.io_core import IOCore
from MachineLearning.Utils.path_utils import PathUtils


class SaveResult(IOCore):
    bandpower_subdir = "Rel_bandpower"
    shannon_entropy_subdir = "Shannon_entropy"
    spectral_skewness_subdir = "Spectral_skewness"
    spectral_kurtosis_subdir = "Spectral_kurtosis"

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
        psd_dir = self.create_psd_path()
        abcd_subdir = PathUtils.create_A_B_C_D_subfolder_name("PSD", parameters)
        xy_subdir = PathUtils.create_X_Y_subfolder_name(parameters)
        psd_dir_fullpath = PathUtils.create_anypath(psd_dir, abcd_subdir, xy_subdir)

        # create directory
        os.makedirs(psd_dir_fullpath, exist_ok=True)

        # create dataframe from PSD data
        psd_df = pd.DataFrame({
            self.psd_freq_col: frequencies,
            self.psd_power_col: power
        })

        # create fullpath with PSD name to save data
        psd_filename = f"PSD_{start}_{end}_{result_id}.csv"
        fullpath = PathUtils.create_anypath(psd_dir_fullpath, psd_filename)

        # save psd
        psd_df.to_csv(fullpath, index=False)

        print(f"Saved: {fullpath}")

    def save_feature_summary_episode(self, results: list, subdir_prefix: str, parameters: dict):
        """
        Saves a csv with all episodes of a given parameter combination
        :param results: list of results
        :param subdir_prefix: prefix of the subdirectory where the summary episode will be saved
        :param parameters: parameters for the episode -> define the subfolder names
        """
        # save as CSV in subfolder with prefix
        result_df = pd.DataFrame(results)
        subdir_of_feature = PathUtils.create_anypath(self.data_dir, self.features_subdir, subdir_prefix)
        abcd_subdir = PathUtils.create_A_B_C_D_subfolder_name(subdir_prefix, parameters)
        subdir_of_file = PathUtils.create_anypath(subdir_of_feature, abcd_subdir)
        fullpath = PathUtils.create_csv_fullpath(subdir_of_feature, subdir_prefix, parameters)
        os.makedirs(subdir_of_file, exist_ok=True)
        result_df.to_csv(fullpath, index=False)  # write without row index

    def save_filtered_eeg(self, filtered_eeg: np.ndarray, fs: int, result_id: int):
        """
        Saves a filtered EEG as a result_id.csv. Assuming the EEG has only 2 channels, these
        will be the columns of the csv-file, and a third column for the fs (sampling frequency)
        :param filtered_eeg: An array of filtered EEG (assuming 2 columns i.e. channels)
        :param fs: The sampling frequency of the EEG
        :param result_id: The patient ID. Will be part of the saved file -> result_id.csv
        """
        # Create a DataFrame and insert fs as the first row
        df = pd.DataFrame(filtered_eeg, columns=['Channel_1', 'Channel_2'])
        df.insert(0, 'fs', '')

        # Add fs value to the first row
        first_row = pd.DataFrame({'fs': [fs], 'Channel_1': [np.nan], 'Channel_2': [np.nan]})
        df = pd.concat([first_row, df], ignore_index=True)

        # Save to CSV
        filtered_eeg_subdir = self.create_filtered_data_path()
        os.makedirs(os.path.dirname(filtered_eeg_subdir), exist_ok=True)
        fullpath = PathUtils.create_anypath(filtered_eeg_subdir, f"{result_id}.csv")
        df.to_csv(fullpath, index=False)
