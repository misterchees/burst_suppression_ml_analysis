import scipy.signal as signal
from MachineLearning.IO.load_data import LoadData
from MachineLearning.IO.save_result import SaveResult


class Filtering:
    loader = LoadData()
    saver = SaveResult()

    def __init__(self):
        pass

    def filter_eeg(self, result_id: int, lowcut, highcut):
        loader = self.loader
        saver = self.saver
        # Extract information from .mat file
        fs, raw_eeg = loader.return_eeg_tuple(result_id)

        # Design Butterworth bandpass filter
        ny_freq = 0.5 * fs
        # Normed cuts on nyquist, because this is a digital filter
        low = lowcut / ny_freq
        high = highcut / ny_freq
        b, a = signal.butter(N=4, Wn=[low, high], btype='band')

        # Apply filter to each channel
        filtered_eeg = signal.filtfilt(b, a, raw_eeg, axis=0)
        saver.save_filtered_eeg(filtered_eeg, fs, result_id)
