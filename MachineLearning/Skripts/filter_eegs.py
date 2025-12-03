from MachineLearning.IO.load_data import LoadData
from MachineLearning.IO.save_result import SaveResult
from MachineLearning.Utils.filter_utils import FilterUtils
from scipy import signal
import pandas as pd

loader = LoadData()
saver = SaveResult()

def filter_all_eegs(lowcut, highcut, order):
    all_ids = loader.return_all_patient_ids("raw_eeg_mat")
    for result_id in all_ids:
        butterworth(result_id, lowcut, highcut, order)


def butterworth(result_id: int, lowcut, highcut, order):
    """
    Applies butterworth bandpass filtering to raw-EEG specified by the result ID and saves the result in
    the filtered subdirectory
    :param result_id: Patient ID. Specifies raw EEG file with name <result_id>.csv
    :param lowcut: Lower bound of the bandpass filter (Hz)
    :param highcut: Upper bound of the bandpass filter (Hz)
    :param order: Order of the bandpass filter -> How steep is the power transition to the filtered frequencies
    """
    # Extract information from .mat file
    print(f"Filtering of EEG from Patient ID: {result_id} in progress")
    fs, raw_eeg = loader.load_eeg_data(result_id=result_id, filtered=False)

    # Design Butterworth bandpass filter
    b, a = FilterUtils.design_butterworth(fs, lowcut=lowcut, highcut=highcut, order=order)

    # Apply filter to each channel
    filtered_eeg = signal.filtfilt(b, a, raw_eeg, axis=0)

    filtered_eeg_df = pd.DataFrame(filtered_eeg, columns=[1,2])
    filtered_eeg_df.to_parquet(f"D:\\Daten\\Filtered_05_40\\{result_id}.parquet", index=False)

    print(f"Patient ID: {result_id} succesfully filtered and saved in filtered_05_40 subdirectory")

if __name__ == "__main__":
    filter_all_eegs(0.5, 40, 4)
