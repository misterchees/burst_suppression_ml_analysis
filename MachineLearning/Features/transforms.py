"""This module contains the Transforms class."""
import numpy as np
from typing import Tuple
from scipy.signal import welch
from MachineLearning.Core.ml_object import MLObject
from MachineLearning.IO.load_data import LoadData
from MachineLearning.IO.save_result import SaveResult, PathUtils


class Transforms(MLObject):
    """
    This class provides methods to transform EEG data from time domain to frequency domain
    i.e. calculate frequency periodograms from linear EEG recordings over time.
    """
    def __init__(self, epoch_types, parameter_kwargs):
        """
        Calls the constructor of the MLObject superclass.
        :param epoch_types: Tuple of all epoch types, that will be handled by this instance.
        :param parameter_kwargs: Dict of parameters to change.
        """
        super().__init__(epoch_types, parameter_kwargs)

    def transform_eeg_episodes_to_psd(self, channel=1, nperseg_seconds=2):
        """
        Calculates PSDs for fake awake EEG windows in specified csv from the pre and saves
        every PSD in a seperate csv file in a defined output directory.

        :param channel: EEG-Channel (options: 1, 2)
        :param nperseg_seconds: Length of window for Welch in seconds (usually: 1 or 2)
        """
        # instances for IO classes to load and save
        result_saver = SaveResult()

        # Update epochs
        self.update_current_epochs(channel)

        # Calculate and save PSDs, depending on epoch type
        for epoch_type in self.epoch_types:
            self.calculate_and_save_psd_for_epochs(nperseg_seconds, epoch_type, result_saver)

    def calculate_and_save_psd_for_epochs(self, nperseg_seconds: int, epoch_type: str, result_saver: SaveResult):
        """
        Calculates and saves PSDs for defined epochs.
        :param nperseg_seconds: Length of window for Welch in seconds (usually: 1 or 2)
        :param epoch_type: Type of epochs from which to calculate PSD.
        :param result_saver: Result Saver instance.
        """
        # Acts as a flag to see if diff to the current psd folder is needed to skip PSD creation
        psd_dir_fullpath = None

        # Define epochs and saving function based on epoch type
        if epoch_type == 'faw':
            epochs = self.faw_epochs.epoch_times
            saving_func = result_saver.save_faw_psd
            # Path to check PSD against current epochs
            psd_dir_fullpath = result_saver.return_all_parameter_fullpath(
                self.parameter_dict, False, False, "features", "psds"
            )

        elif epoch_type == 'awake':
            epochs = self.awake_epochs.epoch_times
            saving_func = result_saver.save_awake_psd
            # Path to check PSD against current epochs
            psd_dir_fullpath = result_saver.return_no_parameters_fullpath(
                self.parameter_dict, "awake",False, "features", "psds"
            )

        elif epoch_type == 'normal_an':
            epochs = self.normal_an_epochs.epoch_times
            saving_func = result_saver.save_normal_an_psd
            # No path given, since the PSD folder should always be cleared to fill with PSDs from new sampled epochs

        else:
            raise ValueError(f'Unrecognized epoch type {epoch_type}. Valid types are "faw", "awake", "normal_an"')

        # Skip PSD calculation if all Epochs are already calculated
        if psd_dir_fullpath is not None:
            if not PathUtils.diff_epochs_vs_psd_files(psd_dir_fullpath, epochs):
                print(f"Skipping Calculations for {epoch_type} epochs. "
                      f"PSDs already calculated for parameters: \n{self.parameter_dict}")
                return

        # Clear folder before calculating new PSDs
        result_saver.clear_psd_folder(self.parameter_dict, epoch_type)

        # Calculate and save psd for each epoch
        for start_time, end_time, result_id, fs, eeg_segment in epochs:
            frequencies, power = self.return_psd(eeg_segment, fs, nperseg_seconds)
            saving_func(frequencies, power, self.parameter_dict, start_time, end_time, result_id)

    def transform_eeg_to_psd(self, result_id: int, channel=1, nperseg_seconds=2):
        """
        Calculates a PSD for a raw EEG from a patient specified by result ID.
        :param result_id: Patient id
        :param channel: EEG-Channel (options: 1, 2)
        :param nperseg_seconds: Length of window for Welch in seconds (usually: 1 or 2)
        """

        # instances for IO classes to load and save
        data_loader = LoadData()
        result_saver = SaveResult()

        # get eeg data for episodes in Patients record (defined by result_id)
        fs, raw_eeg = data_loader.return_eeg_tuple(result_id)

        # validate channel
        if channel not in [1, 2]:
            raise ValueError(f"Channel value is: {channel} but must be 1 or 2")

        eeg_signal = raw_eeg[:, channel - 1]

        # calculate welch PSD
        frequencies, power = self.return_psd(eeg_signal, fs, nperseg_seconds)

        result_saver.save_wholeEEG_psd(frequencies, power, False, result_id)

    @staticmethod
    def return_psd(signal: np.ndarray, fs: int, nperseg_seconds=2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates a PSD from given signal with given parameters.
        :param signal: EEG signal to be transformed
        :param fs: sampling rate in Hz
        :param nperseg_seconds: Length of window for Welch in seconds
        :return: Tuple of frequencies and respective power
        """

        # calculate welch PSD
        nperseg = int(nperseg_seconds * fs)
        frequencies, power = welch(signal, fs=fs, nperseg=nperseg)
        return frequencies, power
