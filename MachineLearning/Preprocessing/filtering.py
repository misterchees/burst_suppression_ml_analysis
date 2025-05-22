import scipy.io
import scipy.signal as signal
import pandas as pd
import numpy as np
import os

class FilterFunctions:
    # directory of all preprocessed csv data
    preprocessing_dir = "C:\\Users\\jesus\\OneDrive\\Dokumente\\Jesús\\Studium\\Fächer - Bioinformatik\\Praktische Arbeit und Bachelorarbeit\\Material\\Daten\\Preprocessing"
    # directory of raw EEGs as .mat data
    vitaldb_eeg_dir = "C:\\Users\\jesus\\OneDrive\\Dokumente\\Jesús\\Studium\\Fächer - Bioinformatik\\Praktische Arbeit und Bachelorarbeit\\Material\\Daten\\Initial data\\vitalDB_mat_EEG"
    # output directory for current calculated feature
    output_dir = "C:\\Users\\jesus\\OneDrive\\Dokumente\\Jesús\\Studium\\Fächer - Bioinformatik\\Praktische Arbeit und Bachelorarbeit\\Material\\Daten\\Features"

    def __init__(self, **kwargs):
        """
        Initialize the EEGFeatureExtractor with optional variables. It checks for every attribute in
        EEGFeatureExtractor and uses the initialized default if no value is given in the kwargs.
        """
        for attr in ["preprocessing_dir", "vitaldb_eeg_dir", "output_dir"]:
            setattr(self, attr, kwargs.get(attr, getattr(self.__class__, attr)))

    def filter_and_save_eeg(self, mat_file_path, output_csv_path, lowcut, highcut):
        # Load .mat file
        mat = scipy.io.loadmat(mat_file_path)

        fs = float(mat['fs'].squeeze())  # Sampling rate
        raw_eeg = mat['rawEEG']  # EEG data (columns 1 and 2)

        # Design Butterworth bandpass filter
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(N=4, Wn=[low, high], btype='band')

        # Apply filter to each channel
        filtered_eeg = signal.filtfilt(b, a, raw_eeg, axis=0)

        # Create a DataFrame and insert fs as the first row
        df = pd.DataFrame(filtered_eeg, columns=['Channel_1', 'Channel_2'])
        df.insert(0, 'fs', '')

        # Add fs value to the first row
        first_row = pd.DataFrame({'fs': [fs], 'Channel_1': [np.nan], 'Channel_2': [np.nan]})
        df = pd.concat([first_row, df], ignore_index=True)

        # Save to CSV
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False)

        return df.head()
