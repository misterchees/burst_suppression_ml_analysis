import os

from MachineLearning.Preprocessing.normalizing import Normalizing
from MachineLearning.IO.load_data import LoadData
from MachineLearning.IO.save_result import SaveResult
from MachineLearning.Utils.filter_utils import FilterUtils
from MachineLearning.Preprocessing.filtering import Filtering
from scipy import signal
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

loader = LoadData()
saver = SaveResult()
filter_instance = Filtering("butterworth")
normalize_instance = Normalizing("zscore")
path_csv = r"E:\Daten\Initial_data\Cleaned_awake_EEG"
path_mat = r"E:\Daten\Initial_data\vitalDB_mat_EEG"

def preprocess_all_eegs(lowcut, highcut, order):
    # Ensure output directory exists to avoid crashes
    output_dir = r"E:\Daten\trimmed_normalized_filtered_05_40"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_ids = loader.return_all_patient_ids("raw_eeg_mat")
    for result_id in all_ids:
        # Load data of patient
        try:
            fs, raw_eeg = loader.load_eeg_data(result_id=result_id, filtered=False)
        except Exception as e:
            print(f"Error loading ID {result_id}: {e}")
            continue
        print(f"Processing of EEG from Patient ID: {result_id} in progress")

        # Trim eeg
        trimmed_eeg = trimming_zeroes(raw_eeg, result_id)

        # Safety Check: If trimming resulted in empty array (or was empty to begin with)
        if trimmed_eeg.size == 0:
            print(f"Skipping Patient {result_id}: Data empty after trimming.")
            continue

        normalized_eeg = trimmed_eeg - np.mean(trimmed_eeg, axis=0)
        butterworth(normalized_eeg, fs, lowcut, highcut, order, result_id, output_dir)


def trimming_zeroes(raw_eeg, result_id):
    # Check along axis 1 (channels). If any channel has a non-zero value, the row is considered valid.
    has_data_mask = np.any(raw_eeg != 0, axis=1)

    if np.any(has_data_mask):
        # np.argmax returns the index of the first True value
        first_idx = np.argmax(has_data_mask)

        # To find the last index, we reverse the array, find the first True, and subtract from length
        last_idx = len(has_data_mask) - np.argmax(has_data_mask[::-1])

        # Apply the slice to remove leading/trailing zeros
        # We print info to see how much was removed
        print(f"Trimming zeros: Cut {first_idx} samples from start and {len(has_data_mask) - last_idx} from end.")
        return raw_eeg[first_idx: last_idx]
    else:
        print(f"Warning: Patient {result_id} contains only zeros. No trimming performed.")
        return np.array([]).reshape(0, raw_eeg.shape[1])


def butterworth(eeg_data, fs, lowcut, highcut, order, result_id, output_dir):
    """
    Applies butterworth bandpass filtering to raw-EEG specified by the result ID and saves the result in
    the filtered subdirectory
    :param eeg_data: Raw EEG signal.
    :param fs: Sampling frequency.
    :param lowcut: Lower bound of the bandpass filter (Hz)
    :param highcut: Upper bound of the bandpass filter (Hz)
    :param order: Order of the bandpass filter -> How steep is the power transition to the filtered frequencies
    :param result_id: Patient ID.
    :param output_dir: Output directory.
    """

    # Design Butterworth bandpass filter
    b, a = FilterUtils.design_butterworth(fs, lowcut=lowcut, highcut=highcut, order=order)

    # Apply filter to each channel
    filtered_eeg = signal.filtfilt(b, a, eeg_data, axis=0)

    # Convert to float32 to save 50% storage space while maintaining sufficient precision for EEG
    filtered_eeg = filtered_eeg.astype(np.float32)

    # Create DataFrame
    filtered_eeg_df = pd.DataFrame(filtered_eeg, columns=[1, 2])

    save_path = os.path.join(output_dir, f"{result_id}.parquet")
    filtered_eeg_df.to_parquet(save_path, index=False)

    print(f"Patient ID: {result_id} successfully filtered and saved to {save_path}")


def filter_cleaned_awake_eegs():
    try:
        files_to_filter = find_missing_mat_equivalents(path_csv, path_mat)

        print(f"Found {len(files_to_filter)} CSV files without a MAT equivalent.")
        print(f" IDs: {files_to_filter}")

    except FileNotFoundError as e:
        print(f"Error: {e}")

    for f in files_to_filter:
        filter_instance.butterworth(int(f), 0.5, 30, 4)

def normalize_cleaned_awake_eegs():
    try:
        files_to_normalize = find_missing_mat_equivalents(path_csv, path_mat)

        print(f"Found {len(files_to_normalize)} CSV files without a MAT equivalent.")
        print(f" IDs: {files_to_normalize}")

    except FileNotFoundError as e:
        print(f"Error: {e}")

    # conversion to int
    files_to_normalize = [int(file_id) for file_id in files_to_normalize]
    normalize_instance.normalize_multiple_eeg(eeg_list=files_to_normalize)




def find_missing_mat_equivalents(csv_source_dir: str, mat_target_dir: str) -> List[str]:
    """
    Identifies file IDs that exist as .csv in the source directory but are missing
    as .mat files in the target directory.

    :param csv_source_dir: Directory path containing the source .csv files.
    :param mat_target_dir: Directory path containing the .mat files to check against.
    :return: A sorted list of IDs (strings) that have no corresponding .mat file.
    """
    source_path = Path(csv_source_dir)
    target_path = Path(mat_target_dir)

    # Check if directories actually exist to prevent confusion
    if not source_path.exists() or not target_path.exists():
        raise FileNotFoundError("One or both of the provided directories do not exist.")

    # 1. Collect all IDs from the CSV directory
    # f.stem extracts the filename without the extension (e.g., '123' from '123.csv')
    # We use a set comprehension for efficiency and to perform set operations later
    csv_ids = {f.stem for f in source_path.glob('*.csv')}

    # 2. Collect all IDs from the MAT directory
    mat_ids = {f.stem for f in target_path.glob('*.mat')}

    # 3. Calculate the difference
    # 'missing_ids' will contain items that are in csv_ids but NOT in mat_ids
    missing_ids = csv_ids - mat_ids

    # Return as a sorted list for easier reading/processing
    return sorted(list(missing_ids))




if __name__ == "__main__":
    # preprocess_all_eegs(0.5, 40, 4)
    # filter_cleaned_awake_eegs()
    normalize_cleaned_awake_eegs()
