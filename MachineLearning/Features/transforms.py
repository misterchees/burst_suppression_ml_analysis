"""This module contains the Transforms class."""
import numpy as np
from typing import Tuple
from scipy.signal import welch
from MachineLearning.Core.ml_object import MLObject
from MachineLearning.IO.load_data import LoadData
from MachineLearning.IO.save_result import SaveResult
from MachineLearning.Utils.config_handler import load_config


class Transforms(MLObject):
    """
    This class provides methods to transform EEG data from time domain to frequency domain
    i.e. calculate frequency periodograms from linear EEG recordings over time.
    """
    def __init__(self, epoch_types, transform_method, parameter_kwargs):
        """
        Calls the constructor of the MLObject superclass.
        :param epoch_types: Tuple of all epoch types, that will be handled by this instance.
        :param transform_method: Name of the transform method to be used.
        :param parameter_kwargs: Dict of parameters to change.
        """
        super().__init__(epoch_types, parameter_kwargs)
        self.transform_method = transform_method

    def transform_eeg_episodes_to_psd(self):
        """
        Calculates PSDs for fake awake EEG windows in specified csv from the pre and saves
        every PSD in a seperate csv file in a defined output directory.
        """
        # instances for IO classes to load and save
        result_saver = SaveResult()

        # Get channel and update epochs
        channel = load_config("parameters_config.yaml")["transform_params"][self.transform_method]["channel"]
        self.update_current_epochs(channel)

        # Calculate and save PSDs, depending on epoch type
        for epoch_type in self.epoch_types:
            self.calculate_and_save_psd_for_epochs(epoch_type, result_saver)

    def calculate_and_save_psd_for_epochs(self, epoch_type: str, result_saver: SaveResult):
        """
        Calculates and saves PSDs for defined epochs.
        :param epoch_type: Type of epochs from which to calculate PSD.
        :param result_saver: Result Saver instance.
        """
        # Define epochs and saving function based on epoch type
        if epoch_type == 'faw':
            epochs = self.faw_epochs.epoch_times
            saving_func = result_saver.save_faw_psd

        elif epoch_type == 'awake':
            epochs = self.awake_epochs.epoch_times
            saving_func = result_saver.save_awake_psd

        elif epoch_type == 'normal_an':
            epochs = self.normal_an_epochs.epoch_times
            saving_func = result_saver.save_normal_an_psd

        else:
            raise ValueError(f'Unrecognized epoch type {epoch_type}. Valid types are "faw", "awake", "normal_an"')

        # Clear folder before calculating new PSDs
        result_saver.clear_psd_folder(self.parameter_dict, epoch_type)

        # Calculate and save psd for each epoch
        for start_time, end_time, result_id, fs, eeg_segment in epochs:
            frequencies, power = self.calculate_psd(eeg_segment)
            saving_func(frequencies, power, self.parameter_dict, start_time, end_time, result_id)

    def calculate_psd(self, eeg_segment: np.ndarray):
        """
        Calculates the Power Spectral Density (PSD) of the provided EEG data segment using a specified
        transformation method. This method currently supports the Welch method for PSD calculation.

        :param eeg_segment: A numpy array representing the EEG segment for which the Power
           Spectral Density needs to be calculated. The array should contain time-series EEG data.
        :type eeg_segment: numpy.ndarray
        :returns:
        - frequencies (numpy.ndarray) -- Array of frequency bins.
        - power (numpy.ndarray) -- Corresponding power values for each frequency bin.
        :rtype: tuple(numpy.ndarray, numpy.ndarray)
        :raises ValueError: If the specified transform method is unrecognized or not supported.
        """
        if self.transform_method == "welch":
            params = load_config("parameters_config.yaml")["transform_params"][self.transform_method]
            fs = params["fs"]
            nperseg_seconds = params["nperseg_seconds"]

            frequencies, power = self.calculate_psd_welch(eeg_segment, fs, nperseg_seconds)
            return frequencies, power

        else:
            raise ValueError(f"Unrecognized transform method: {self.transform_method}")

    @staticmethod
    def calculate_psd_welch(signal: np.ndarray, fs: int, nperseg_seconds: int) -> Tuple[np.ndarray, np.ndarray]:
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

    def transform_eeg_to_psd_welch(self, result_id: int, channel: int, nperseg_seconds: int):
        """
        Calculates a PSD for the whole length of a raw EEG from a patient specified by result ID.
        :param result_id: Patient id
        :param channel: EEG-Channel (options: 1, 2)
        :param nperseg_seconds: Length of window for Welch in seconds (usually: 1 or 2)
        """

        # instances for IO classes to load and save
        data_loader = LoadData()
        result_saver = SaveResult()

        # get eeg data for episodes in Patients record (defined by result_id)
        fs, raw_eeg = data_loader.load_eeg_data(result_id)

        # validate channel
        if channel not in [1, 2]:
            raise ValueError(f"Channel value is: {channel} but must be 1 or 2")

        eeg_signal = raw_eeg[:, channel - 1]

        # calculate welch PSD
        frequencies, power = self.calculate_psd_welch(eeg_signal, fs, nperseg_seconds)

        result_saver.save_complete_eeg_psd(frequencies, power, False, result_id)