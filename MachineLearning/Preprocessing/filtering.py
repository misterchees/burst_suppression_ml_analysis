import scipy.signal as signal
from MachineLearning.IO.load_data import LoadData
from MachineLearning.IO.save_result import SaveResult
from MachineLearning.Utils.filter_utils import FilterUtils


class Filtering:
    loader = LoadData()
    saver = SaveResult()

    def __init__(self):
        pass

    def butterworth(self, result_id: int, lowcut=0.5, highcut=30.0, order=4):
        """
        Applies butterworth bandpass filtering to raw-EEG specified by the result ID and saves the result in
        the filtered subdirectory
        :param result_id: Patient ID. Specifies raw EEG file with name <result_id>.csv
        :param lowcut: Lower bound of the bandpass filter (Hz)
        :param highcut: Upper bound of the bandpass filter (Hz)
        :param order: Order of the bandpass filter -> How steep is the power transition to the filtered frequencies
        """
        loader = self.loader
        saver = self.saver
        # Extract information from .mat file
        print(f"Filtering of EEG from Patient ID: {result_id} in progress")
        fs, raw_eeg = loader.return_eeg_tuple(result_id=result_id, filtered=False)

        # Design Butterworth bandpass filter
        b, a = FilterUtils.design_butterworth(fs, lowcut=lowcut, highcut=highcut, order=order)

        # Apply filter to each channel
        filtered_eeg = signal.filtfilt(b, a, raw_eeg, axis=0)
        saver.save_filtered_eeg(filtered_eeg, fs, result_id)
