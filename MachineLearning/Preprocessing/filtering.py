import scipy.signal as signal
from MachineLearning.IO.load_data import LoadData
from MachineLearning.IO.save_result import SaveResult


class Filtering:
    loader = LoadData()
    saver = SaveResult()

    def __init__(self):
        pass

    def butterworth(self, result_id: int, lowcut=0.5, highcut=30.0, order=500):
        """
        Applies butterworth bandpass filtering to raw-EEG specified by the result ID and saves the result in
        the filtered subdirectory
        :param result_id: Patient ID. Specifies raw EEG file with name <result_id>.csv
        :param lowcut: Lower bound of the bandpass filter
        :param highcut: Upper bound of the bandpass filter
        :param order: Order of the bandpass filter -> How steep is the power transition to the filtered frequencies
        """
        loader = self.loader
        saver = self.saver
        # Extract information from .mat file
        fs, raw_eeg = loader.return_eeg_tuple(result_id)

        # Design Butterworth bandpass filter
        ny_freq = 0.5 * fs
        # Normed cuts on nyquist, because this is a digital filter
        low = lowcut / ny_freq
        high = highcut / ny_freq
        b, a = signal.butter(N=order, Wn=[low, high], btype='band')

        # Apply filter to each channel
        filtered_eeg = signal.filtfilt(b, a, raw_eeg, axis=0)
        saver.save_filtered_eeg(filtered_eeg, fs, result_id)
