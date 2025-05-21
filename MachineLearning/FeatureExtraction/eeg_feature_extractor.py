import os
import warnings
import pandas as pd
import scipy.io
from scipy.signal import welch
import numpy as np


class EEGFeatureExtractor:
    # directory of all preprocessed csv data
    preprocessing_dir = "C:\\Users\\jesus\\OneDrive\\Dokumente\\Jesús\\Studium\\Fächer - Bioinformatik\\Praktische Arbeit und Bachelorarbeit\\Material\\Daten\\Preprocessing"
    # directory of raw EEGs as .mat data
    vitaldb_eeg_dir = "C:\\Users\\jesus\\OneDrive\\Dokumente\\Jesús\\Studium\\Fächer - Bioinformatik\\Praktische Arbeit und Bachelorarbeit\\Material\\Daten\\Initial data\\vitalDB_mat_EEG"
    # output directory for current calculated feature
    output_dir = "C:\\Users\\jesus\\OneDrive\\Dokumente\\Jesús\\Studium\\Fächer - Bioinformatik\\Praktische Arbeit und Bachelorarbeit\\Material\\Daten\\Features"

    merged_episodes = False  # flag to determine if episodes are merged
    bis_threshold = 70  # lower threshold on BIS value (options: 70)
    mac_threshold = 0.8  # lower threshold on MAC value (options: 0.5, 0.6, 0.7, 0.8)
    min_episode_length = 20  # lower threshold on episode length (options: 5, 6, 7, 8, 9, 10, 15, 20)
    refractory_time = 5  # maximum refractory time between episodes in seconds (options: 3, 4, 5)
    fixed_window_size = 20  # exact window length (options: 5, 6, 7, 8, 9, 10, 15, 20)
    overlap = 0.0  # window overlap (options: 0.0, 0.25, 0.5)

    eeg_fs = "fs"
    eeg_rawEEG = "rawEEG"

    psd_freq_col = "Frequency_Hz"
    psd_power_col = "PSD_V2_per_Hz"

    # Typical bands of EEG
    frequency_bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
        "Gamma": (30, 45)
    }

    def __init__(self, **kwargs):
        """
        Initialize the EEGFeatureExtractor with optional variables. It checks for every attribute in
        EEGFeatureExtractor and uses the initialized default if no value is given in the kwargs.
        """
        for attr in ["preprocessing_dir", "vitaldb_eeg_dir", "output_dir", "merged_episodes", "bis_threshold",
                     "mac_threshold", "min_episode_length", "refractory_time", "fixed_window_size", "overlap"]:
            setattr(self, attr, kwargs.get(attr, getattr(self.__class__, attr)))

    def extract_psd(self, channel=1, nperseg_seconds=2):
        """
        Calculates PSDs for EEG windows in specified csv from preprocessing_csv_fullpath and saves
        every PSD in a seperate csv file in a defined output directory.

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
        psd_output_path = os.path.join(self.output_dir, "PSDs", psd_subfolder_1, psd_subfolder_2)
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
            fs = int(mat_data[self.eeg_fs].squeeze())
            raw_eeg = mat_data[self.eeg_rawEEG]

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
                self.psd_freq_col: frequencies,
                self.psd_power_col: psd
            })

            # save as PSD_H_K_L.csv according to structure in preprocessing CSV file. i.e. start, end, resultID
            psd_filename = f"PSD_{start_time}_{end_time}_{result_id}.csv"
            psd_file_path = os.path.join(psd_output_path, psd_filename)
            psd_df.to_csv(psd_file_path, index=False)

            print(f"Saved: {psd_file_path}")

    def extract_relative_bandpower(self):
        """
        Uses the function calculate_relative_bandpower() to calculate the relative band power in all
        windows of the PSD directory.
        """

        # create path to directory with PSDs (specified by class attributes)
        psd_subfolder_1 = self.create_A_B_C_D_subfolder_name("PSD")
        psd_subfolder_2 = self.create_X_Y_subfolder_name()
        psd_directory_path = os.path.join(self.output_dir,"PSDs", psd_subfolder_1, psd_subfolder_2)

        # Output
        all_rows = []

        for psd_window_file in os.listdir(psd_directory_path):
            if psd_window_file.endswith(".csv"):
                psd_fullpath = os.path.join(psd_directory_path, psd_window_file)
                print(f"Processing {psd_fullpath}")
                metadata = psd_window_file.replace(".csv", "").split("_")[1:]   # PSD_0_1_2.csv -> ['0','1','2']

                psd_dataframe = pd.read_csv(psd_fullpath)
                relative_bandpowers = self.calculate_relative_bandpower(psd_dataframe)
                row = {
                    "ResultID": int(metadata[0]),
                    "Start": int(metadata[1]),
                    "End": int(metadata[2]),      # 2.csv -> 2
                    **relative_bandpowers  # unpack dict with band powers
                }
                all_rows.append(row)

        # save as CSV in bandpower subfolder
        result_df = pd.DataFrame(all_rows)
        output_dir = os.path.join(self.output_dir, "rel_bandpowers", self.create_A_B_C_D_subfolder_name("RelBandpower"))
        os.makedirs(output_dir, exist_ok=True)
        result_df.to_csv(os.path.join(output_dir, f"{psd_subfolder_2}.csv"), index=False)  # write without row index

    def calculate_relative_bandpower(self, psd_df: pd.DataFrame, normalize_to="bands"):
        """
        Calculate relative bandpower from PSD DataFrame.

        :param psd_df: PSD DataFrame with columns 'Frequency_Hz' and 'PSD_V2_per_Hz'
        :param normalize_to: 'total' → normalize to total PSD power (default);
                            'bands' → normalize only to sum of power in specified bands
        :return: dict of relative power values per band
        """

        freqs = psd_df[self.psd_freq_col].values
        power = psd_df[self.psd_power_col].values

        # Compute total power for denominator depending on strategy
        if normalize_to == "total":
            total_power = np.trapezoid(power, freqs)    # estimate total power (i.e. AUC) with trapezoid rule
        elif normalize_to == "bands":
            # Only sum the power within all band ranges
            mask = np.zeros_like(freqs, dtype=bool)
            for low, high in self.frequency_bands.values():
                mask |= (freqs >= low) & (freqs < high)     # bitwise OR to use all band values or zeros
            total_power = np.trapezoid(power[mask], freqs[mask])
        else:
            raise ValueError("normalize_to must be either 'total' or 'bands'")

        result = {}
        for band_name, (low, high) in self.frequency_bands.items():  # for each frequency band (specified in class)
            mask = (psd_df[self.psd_freq_col] >= low) & (psd_df[self.psd_freq_col] < high)  # gather frequencies of band
            band_power = np.trapezoid(psd_df.loc[mask, self.psd_power_col], psd_df.loc[mask, self.psd_freq_col])
            relative_power = band_power / total_power if total_power > 0 else 0
            result[band_name] = relative_power
        return result

    def extract_shannon_entropy(self):
        """
        Iterates over all PSD CSVs in the Feature PSD folder, calculates Shannon entropy,
        and saves the results as CSVs in an shannonEntropy output folder.
        """

        output_dir = os.path.join(self.output_dir, "ShannonEntropies", self.create_A_B_C_D_subfolder_name("ShannonEntropy"))
        os.makedirs(output_dir, exist_ok=True)  # Create output folder if needed

        psd_input_dir = os.path.join(self.output_dir, "PSDs",
                                     self.create_A_B_C_D_subfolder_name("PSD"), self.create_X_Y_subfolder_name())   # Create path to PSDs

        all_rows = []

        for filename in os.listdir(psd_input_dir):
            if filename.endswith(".csv"):
                psd_fullpath = os.path.join(psd_input_dir, filename)
                print(f"Processing {psd_fullpath}")
                psd_df = pd.read_csv(psd_fullpath)

                # Calculate entropy
                entropy = self.calculate_shannon_entropy(psd_df)

                # Extract metadata from filename
                parts = filename.replace(".csv", "").split("_")  # ['PSD', 'ResultID', 'Start', 'End']
                if len(parts) != 4:
                    warnings.warn(f"skipped due to unexpected filename format: {filename}")
                    continue

                result_id = parts[1]
                start = parts[2]
                end = parts[3]

                # Create result row
                result_row = {
                    "ResultID": result_id,
                    "Start": start,
                    "End": end,
                    "ShannonEntropy": entropy
                }
                all_rows.append(result_row)

        result = pd.DataFrame(all_rows)
        # Output filename and path
        out_filename = f"{self.create_X_Y_subfolder_name()}.csv"
        out_path = os.path.join(output_dir, out_filename)

        # Save without index column
        result.to_csv(out_path, index=False)

    def calculate_shannon_entropy(self, psd_df: pd.DataFrame, normalize=True) -> float:
        """
        Calculates the Shannon entropy of a PSD.

        :param normalize: Normalize Entropy (default: True)
        :param psd_df: DataFrame with a 'PSD_V2_per_Hz' column
        :return: Shannon entropy as float
        """

        power = psd_df[self.psd_power_col].values
        total_power = np.sum(power)

        if total_power == 0:
            return np.nan  # Avoid division by zero

        # Normalize PSD to create a probability distribution
        probabilities = power / total_power

        # Avoid log(0) by masking zero entries i.e remove them in nonzero_probs
        nonzero_probs = probabilities[probabilities > 0]

        entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs))

        if normalize and len(nonzero_probs) > 1:
            entropy /= np.log2(len(nonzero_probs))

        return entropy

    def calculate_spectral_skewness(self, psd_df: pd.DataFrame, normalize="0-1") -> float:
        """
        Calculates spectral skewness from a power spectral density (PSD).

        :param psd_df: DataFrame with columns 'Freq_Hz' and 'PSD_V2_per_Hz'
        :param normalize: False -> raw, "tanh" -> confining smoothly to [-1,1] , or "0-1" -> clipped to [0,1] (default)
        :return skewness: Float (normalized if requested)
        """

        freqs = psd_df[self.psd_freq_col].values
        power = psd_df[self.psd_power_col].values
        total_power = np.sum(power)

        if total_power == 0:
            return np.nan

        # Normalize power to form a probability distribution
        probabilities = power / total_power
        # Weighted mean frequency
        mean_freq = np.sum(probabilities * freqs)
        # Weighted standard deviation
        std_freq = np.sqrt(np.sum(probabilities * (freqs - mean_freq) ** 2))

        # no standard deviation -> perfectly symmetric -> no skewness
        if std_freq == 0:
            return 0.5 if normalize == "0-1" else 0.0

        skewness = np.sum(probabilities * ((freqs - mean_freq) / std_freq) ** 3)

        if normalize == "tanh":
            skewness = np.tanh(skewness)
        elif normalize == "0-1":
            skewness = np.clip(skewness, -1.0, 1.0)
            skewness = (skewness + 1) / 2

        return skewness

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
