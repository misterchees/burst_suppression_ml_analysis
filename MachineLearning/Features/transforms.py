import numpy as np
from typing import Tuple
from MachineLearning.Core.ml_object import MLObject
from MachineLearning.IO.load_data import LoadData
from MachineLearning.IO.save_result import SaveResult
from scipy.signal import welch


class Transforms(MLObject):

    def __init__(self, faw: bool, awake: bool):
        super().__init__(faw, awake)

    def transform_eeg_episodes_to_psd(self, channel=1, nperseg_seconds=2, filtered=True):
        """
        Calculates PSDs for fake awake EEG windows in specified csv from the pre and saves
        every PSD in a seperate csv file in a defined output directory.

        :param filtered: Defines if windows are from filtered EEG (True) or raw EEG (False).
        :param channel: EEG-Channel (options: 1, 2)
        :param nperseg_seconds: Length of window for Welch in seconds (usually: 1 or 2)
        """
        # instances for IO classes to load and save
        result_saver = SaveResult()

        # Update epochs
        self.update_current_epochs(channel)

        if self.faw:
            for start_time, end_time, result_id, fs, eeg_segment in self.faw_epochs.epoch_times:
                # calculate welch PSD
                frequencies, power = self.return_psd(eeg_segment, fs, nperseg_seconds)

                result_saver.save_faw_psd(frequencies, power, self.parameter_dict, start_time, end_time, result_id)
        if self.awake:
            for start_time, end_time, result_id, fs, eeg_segment in self.awake_epochs.epoch_times:
                # calculate welch PSD
                frequencies, power = self.return_psd(eeg_segment, fs, nperseg_seconds)

                result_saver.save_awake_psd(frequencies, power, self.parameter_dict, start_time, end_time, result_id)

    def transform_eeg_to_psd(self, result_id: int, channel=1, nperseg_seconds=2):
        """
        Calculates a PSD for a raw EEG from a patient specified by result id .
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
