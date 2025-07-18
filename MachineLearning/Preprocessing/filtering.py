import scipy.signal as signal
from MachineLearning.IO.load_data import LoadData
from MachineLearning.IO.save_result import SaveResult
from MachineLearning.Utils.filter_utils import FilterUtils


class Filtering:

    def __init__(self, method: str, filter_params: dict):
        """
        Represents a class for initializing with a method name and its corresponding
        filter parameters in the form of a dictionary.

        This class is used to specify a method and its respective configuration
        parameters to perform a specific operation or filtering.

        :param method: The name of the method to be initialized.
        :type method: str
        :param filter_params: A dictionary containing the filtering parameters
            for the method.
        :type filter_params: dict
        """
        self.method = method
        self.filter_params = filter_params

    def filter_multiple_eeg(self, eeg_list: list):
        """
        Applies a filter function to all raw-EEGs specified by the result IDs in a given list and saves
        the result in the filtered subdirectory.
        :param eeg_list: list with all result_ids of EEGs to be filtered
        """
        if self.method == "butterworth":
            lowcut = self.filter_params["lowcut"]
            highcut = self.filter_params["highcut"]
            order = self.filter_params["order"]

            for result_id in eeg_list:
                self.butterworth(result_id, lowcut, highcut, order)

        else:
            raise ValueError(f"Unrecognized filter method: {self.method}")

    @staticmethod
    def butterworth(result_id: int, lowcut, highcut, order):
        """
        Applies butterworth bandpass filtering to raw-EEG specified by the result ID and saves the result in
        the filtered subdirectory
        :param result_id: Patient ID. Specifies raw EEG file with name <result_id>.csv
        :param lowcut: Lower bound of the bandpass filter (Hz)
        :param highcut: Upper bound of the bandpass filter (Hz)
        :param order: Order of the bandpass filter -> How steep is the power transition to the filtered frequencies
        """
        loader = LoadData()
        saver = SaveResult()

        # Extract information from .mat file
        print(f"Filtering of EEG from Patient ID: {result_id} in progress")
        fs, raw_eeg = loader.return_eeg_tuple(result_id=result_id, filtered=False)

        # Design Butterworth bandpass filter
        b, a = FilterUtils.design_butterworth(fs, lowcut=lowcut, highcut=highcut, order=order)

        # Apply filter to each channel
        filtered_eeg = signal.filtfilt(b, a, raw_eeg, axis=0)
        saver.save_filtered_eeg(filtered_eeg, fs, result_id)
        print(f"Patient ID: {result_id} succesfully filtered and saved in filtered subdirectory")
