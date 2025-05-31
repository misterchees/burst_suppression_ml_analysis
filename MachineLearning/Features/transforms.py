from MachineLearning.Core.ml_object import MLObject
from MachineLearning.IO.load_data import LoadData
from MachineLearning.IO.save_result import SaveResult
from scipy.signal import welch


class Transform(MLObject):

    def __init__(self):
        super().__init__()

    def transform_eeg_episodes_to_psd(self, channel=1, nperseg_seconds=2, filtered=True):
        """
        Calculates PSDs for fake awake EEG windows in specified csv from the pre and saves
        every PSD in a seperate csv file in a defined output directory.

        :param filtered: Defines if windows are from filtered EEG (True) or raw EEG (False).
        :param channel: EEG-Channel (options: 1, 2)
        :param nperseg_seconds: Length of window for Welch in seconds (usually: 1 or 2)
        """
        # instances for IO classes to load and save
        data_loader = LoadData()
        result_saver = SaveResult()

        # Load FAW Episode based on current parameters
        episode_times_df = data_loader.load_faw_csv_as_df(self.parameter_dict)

        for _, row in episode_times_df.iterrows():
            result_id = int(row['ResultID'])
            start_time = int(row['Start'])
            end_time = int(row['End'])

            # get eeg data for episodes in Patients record (defined by result_id)
            fs, raw_eeg = data_loader.return_eeg_tuple(result_id, filtered)

            # validate channel
            if channel not in [1, 2]:
                raise ValueError(f"Channel value is: {channel} but must be 1 or 2")

            eeg_signal = raw_eeg[:, channel - 1]

            # timeframe in samples
            start_sample = int(start_time * fs)
            end_sample = int(end_time * fs)
            eeg_segment = eeg_signal[start_sample:end_sample]

            # calculate welch PSD
            nperseg = int(nperseg_seconds * fs)
            frequencies, power = welch(eeg_segment, fs=fs, nperseg=nperseg)

            result_saver.save_psd(frequencies, power, self.parameter_dict, start_time, end_time, result_id)

    def transform_eeg_to_psd(self, result_id: int, channel=1, nperseg_seconds=2):
        """
        Calculates a PSDs for a raw EEG from a patient specified by result id .
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
        nperseg = int(nperseg_seconds * fs)
        frequencies, power = welch(eeg_signal, fs=fs, nperseg=nperseg)

        result_saver.save_wholeEEG_psd(frequencies, power, False, result_id)
