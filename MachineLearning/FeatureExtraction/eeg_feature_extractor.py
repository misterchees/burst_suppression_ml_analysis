import os
import warnings
import pandas as pd
import scipy.io
from scipy.signal import welch


class EEGFeatureExtractor:
    def __init__(self, preprocessing_dir="C:\\Users\\jesus\\OneDrive\\Dokumente\\Jesús\\Studium\\Fächer - Bioinformatik\\Praktische Arbeit und Bachelorarbeit\\Material\\Daten\\Preprocessing",
                 vitaldb_eeg_dir="C:\\Users\\jesus\\OneDrive\\Dokumente\\Jesús\\Studium\\Fächer - Bioinformatik\\Praktische Arbeit und Bachelorarbeit\\Material\\Daten\\Initial data\\vitalDB_mat_EEG"
                 , output_dir="C:\\Users\\jesus\\OneDrive\\Dokumente\\Jesús\\Studium\\Fächer - Bioinformatik\\Praktische Arbeit und Bachelorarbeit\\Material\\Daten\\Features\\PSDs"):
        """
        Initialize the EEGFeatureExtractor.

        Parameters:
        - preprocessing_dir: Path to the 'preprocessing' folder containing result_A_B_C_D subfolders.
        - vitaldb_dir: Path to the 'vitalDB_mat_EEG' folder containing X.mat files.
        - output_dir: Path to the 'Features' folder where output should be saved.
        """
        self.preprocessing_dir = preprocessing_dir
        self.vitaldb_eeg_dir = vitaldb_eeg_dir
        self.output_dir = output_dir

    def extract_psd(self, channel=1, nperseg_seconds=2):
        """
        Calculates PSDs for EEG windows based on Summary_Episodes in CSV-files.

        Parameters:
        - preprocessing_dir: Path to "preprocessing" (directory with result_A_B_C_D subfolders)
        - vitaldb_dir: Path to "vitalDB_mat_EEG" (contains X.mat data)
        - output_dir: Path to "Features" (directory where PSDs are saved)
        - channel: EEG-Channel (1 or 2)
        - nperseg_seconds: Length of window for Welch in seconds
        """
        for subfolder in os.listdir(self.preprocessing_dir):
            if subfolder.startswith("result_"):
                subfolder_parts = subfolder.split('_')[1:]  # Extract subfolder name structure ['A', 'B', 'C', 'D']
                result_csv_path = os.path.join(self.preprocessing_dir, subfolder,
                                               "Summary_Episodes.csv")  # fullpath to csv-file

                # validate if input file exists
                if not os.path.isfile(result_csv_path):
                    warnings.warn(f"CSV not found: {result_csv_path}")
                    continue

                input_dataframe = pd.read_csv(result_csv_path)

                # create output directory: PSD_A_B_C_D
                psd_subfolder = f"PSD_{'_'.join(subfolder_parts)}"
                psd_output_path = os.path.join(self.output_dir, psd_subfolder)
                os.makedirs(psd_output_path, exist_ok=True)

                for _, row in input_dataframe.iterrows():
                    result_id = int(row['ResultID'])
                    start_time = int(row['Start'])
                    end_time = int(row['End'])

                    mat_file_path = os.path.join(self.vitaldb_eeg_dir, f"{result_id}.mat")
                    if not os.path.isfile(mat_file_path):
                        warnings.warn(f"MAT-file not found: {mat_file_path}")
                        continue

                    # load .mat file
                    mat_data = scipy.io.loadmat(mat_file_path)
                    fs = int(mat_data['fs'].squeeze())
                    raw_eeg = mat_data['rawEEG']

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
                    frequencies, psd = welch(eeg_segment, fs=fs, nperseg=nperseg)

                    # result as DataFrame
                    psd_df = pd.DataFrame({
                        'Frequency_Hz': frequencies,
                        'PSD_V2_per_Hz': psd
                    })

                    # save as PSD_H_K_L.csv
                    psd_filename = f"PSD_{result_id}_{start_time:.2f}_{end_time:.2f}.csv"
                    psd_file_path = os.path.join(psd_output_path, psd_filename)
                    psd_df.to_csv(psd_file_path, index=False)

                    print(f"Gespeichert: {psd_file_path}")

    # Placeholder methods for future features (e.g., Bandpower, Entropy)
    def extract_bandpower(self):
        pass

    def extract_entropy(self):
        pass
