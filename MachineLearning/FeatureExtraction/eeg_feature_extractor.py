import os
import warnings
import pandas as pd
import scipy.io
from scipy.signal import welch


class EEGFeatureExtractor:
    # directory of all preprocessed csv data
    preprocessing_dir = "C:\\Users\\jesus\\OneDrive\\Dokumente\\Jesús\\Studium\\Fächer - Bioinformatik\\Praktische Arbeit und Bachelorarbeit\\Material\\Daten\\Preprocessing"
    # directory of raw EEGs as .mat data
    vitaldb_eeg_dir = "C:\\Users\\jesus\\OneDrive\\Dokumente\\Jesús\\Studium\\Fächer - Bioinformatik\\Praktische Arbeit und Bachelorarbeit\\Material\\Daten\\Initial data\\vitalDB_mat_EEG"
    # output directory for current calculated feature
    output_dir = "C:\\Users\\jesus\\OneDrive\\Dokumente\\Jesús\\Studium\\Fächer - Bioinformatik\\Praktische Arbeit und Bachelorarbeit\\Material\\Daten\\Features\\PSDs"

    merged_episodes = False     # flag to determine if episodes are merged
    bis_threshold = 70          # lower threshold on BIS value (options: 70)
    mac_threshold = 0.8         # lower threshold on MAC value (options: 0.5, 0.6, 0.7, 0.8)
    min_episode_length = 20     # lower threshold on episode length (options: 5, 6, 7, 8, 9, 10, 15, 20)
    refractory_time = 5         # maximum refractory time between episodes in seconds (options: 3, 4, 5)
    fixed_window_size = 20      # exact window length (options: 5, 6, 7, 8, 9, 10, 15, 20)
    overlap = 0.0               # window overlap (options: 0.0, 0.25, 0.5)

    def __init__(self, **kwargs):
        """
        Initialize the EEGFeatureExtractor with optional variables. It checks for every attribute in
        EEGFeatureExtractor and uses the initialized default if no value is given in the kwargs.
        """
        for attr in ["preprocessing_dir", "vitaldb_eeg_dir", "output_dir", "merged_episodes", "bis_threshold",
                     "mac_threshold",  "min_episode_length", "refractory_time", "fixed_window_size", "overlap"]:
            setattr(self, attr, kwargs.get(attr, getattr(self.__class__, attr)))

    def extract_psd(self, channel=1, nperseg_seconds=2):
        """
        Calculates PSDs for EEG windows in specified csv from preprocessing_csv_fullpath.

        Parameters:
        - channel: EEG-Channel (options: 1, 2)
        - nperseg_seconds: Length of window for Welch in seconds (usually: 1 or 2)
        """

        csv_path = self.create_preprocessing_fullpath()
        # validate if input file exists
        if not os.path.isfile(csv_path):
            warnings.warn(f"CSV not found: {csv_path}")

        input_dataframe = pd.read_csv(csv_path)

        # create output directory with same structure as input subfolder: PSD_A_B_C_D\Summary_Episodes_X_Y
        psd_subfolder_1 = self.create_A_B_C_D_subfolder_name("PSD")
        psd_subfolder_2 = self.create_X_Y_subfolder_name()
        psd_output_path = os.path.join(self.output_dir, psd_subfolder_1, psd_subfolder_2)
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

            # save as PSD_H_K_L.csv according to structure in preprocessing CSV file. i.e. start, end, resultID
            psd_filename = f"PSD_{start_time}_{end_time}_{result_id}.csv"
            psd_file_path = os.path.join(psd_output_path, psd_filename)
            psd_df.to_csv(psd_file_path, index=False)

            print(f"Saved: {psd_file_path}")

    # Placeholder methods for future features (e.g., Bandpower, Entropy)
    def extract_bandpower(self):
        pass

    def extract_entropy(self):
        pass

    def set_attributes(self, **kwargs):
        """
        Sets any number of the attributes of the EEGFeatureExtractor.

        :param merged_episodes: flag to determine if episodes are merged (default: False)
        :param bis_threshold: lower threshold on BIS value (options: 70)
        :param mac_threshold: lower threshold on MAC value (options: 0.5, 0.6, 0.7, 0.8)
        :param min_episode_length: lower threshold on episode length (options: 5, 6, 7, 8, 9, 10, 15, 20)
        :param refractory_time: maximum refractory time between episodes in seconds (options: 3, 4, 5)
        :param fixed_window_size: exact window length (options: 5, 6, 7, 8, 9, 10, 15, 20)
        :param overlap: window overlap (options: 0.0, 0.25, 0.5)
        """

        for attr in ["preprocessing_dir", "vitaldb_eeg_dir", "output_dir", "merged_episodes", "bis_threshold",
                     "mac_threshold", "min_episode_length", "refractory_time", "fixed_window_size", "overlap"]:
            setattr(self, attr, kwargs.get(attr, getattr(self.__class__, attr)))

    def create_preprocessing_fullpath(self):
        """
        Sets the preprocessing fullpath variable to "result_..." with the subsequent dots
        as placeholder for the specific path depending on the attributes of EEGFeatureExtractor.
        """
        return self.create_fullpath("result")

    def create_fullpath(self, prefix):
        """
        Calculates and returns a fullpath string variable defined by the class attributes merged_episodes,
        refractory time, min_episode_length, mac_threshold, bis_threshold, overlap and fixed_window_size.
        Example: result_70_080_20_5\\Summary_Episodes_20_000.csv

        Parameters:
        :param prefix: Prefix of the fullpath variable.
        :return: Fullpath string variable.
        """

        # transform parameters to folder name conventions
        episode_name = "Summary_Episodes"
        if self.merged_episodes:
            episode_name = "Summary_Merged_Episodes"

        # leave 2 digits after decimal point and remove it afterward: 0.5 -> 050, 0.25 -> 025 etc.
        overlap = f"{self.overlap:.2f}".replace(".", "")

        # assemble fullpath to CSV
        subfolder_name = self.create_A_B_C_D_subfolder_name(prefix)
        csv_name = f"{episode_name}_{self.fixed_window_size}_{overlap}.csv"  # Summary_Episodes_X_Y
        return os.path.join(self.preprocessing_dir, subfolder_name, csv_name)

    def create_A_B_C_D_subfolder_name(self, prefix) -> str:
        """
        Calculates and returns a subfolder name variable defined by the class attributes,
        refractory time, min_episode_length, mac_threshold and bis_threshold.
        Example: result_70_080_20_5

        Parameters:
        :param prefix: Prefix of the subfolder name variable.
        :return: Subfolder string variable.
        """
        # leave 2 digits after decimal point and remove it afterward: 0.5 -> 050, 0.25 -> 025 etc.
        mac_threshold = f"{self.mac_threshold:.2f}".replace(".", "")
        return f"{prefix}_{self.bis_threshold}_{mac_threshold}_{self.min_episode_length}_{self.refractory_time}"

    def create_X_Y_subfolder_name(self) -> str:
        """
        Calculates and returns a subfolder name variable defined by the class attributes:
        merged_episodes, fixed_window_size and overlap.
        Example: Summary_Episode_20_000

        Parameters:
        :return: Subfolder string variable.
        """

        episode_name = "Summary_Episodes"
        if self.merged_episodes:
            episode_name = "Summary_Merged_Episodes"
        # leave 2 digits after decimal point and remove it afterward: 0.5 -> 050, 0.25 -> 025 etc.
        overlap = f"{self.overlap:.2f}".replace(".", "")
        return f"{episode_name}_{self.fixed_window_size}_{overlap}"
