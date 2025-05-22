import os
import warnings
import pandas as pd
import numpy as np
from MachineLearning.Core.ml_object import MLObject


class EEGFeatureExtractor(MLObject):
    def __init__(self):
        super().__init__()

    def extract_relative_bandpower_for_current_parameters(self):
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
