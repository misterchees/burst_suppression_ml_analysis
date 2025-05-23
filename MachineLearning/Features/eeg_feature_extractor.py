import os
import pandas as pd
import numpy as np
from MachineLearning.Core.ml_object import MLObject
from MachineLearning.IO.save_result import SaveResult
from MachineLearning.IO.io_core import IOCore
from MachineLearning.IO.load_data import LoadData


class EEGFeatureExtractor(MLObject):
    io_instance = IOCore()
    data_loader = LoadData()
    result_saver = SaveResult()

    def __init__(self):
        super().__init__()

    def extract_relative_bandpower_for_parameter_combination(self):
        """
        Iterates over all PSD CSVs in the Feature PSD folder,calculates the relative band power
        and saves the results as CSV in a Rel_bandpower output folder of the current parameter combination.
        """
        loader = self.data_loader
        saver = self.result_saver
        # create path to directory with PSDs (specified by class attributes)
        psd_dir = loader.create_psd_path_with_parameters(self.parameter_dict)

        # Output
        all_rows = []

        for psd_file in os.listdir(psd_dir):
            if psd_file.endswith(".csv"):
                psd_df, start, end, result_id = loader.load_psd_with_start_end_resultid(psd_dir, psd_file)
                relative_bandpowers = self.calculate_relative_bandpower(psd_df)
                row = {
                    "Start": start,
                    "End": end,
                    "ResultID": result_id,
                    **relative_bandpowers  # unpack dict with band powers
                }
                all_rows.append(row)

        # save as CSV in bandpower subfolder
        saver.save_feature_summary_episode(all_rows, saver.bandpower_subdir, self.parameter_dict)
        print(f"Succesfully calculated and saved bandpowers")

    def calculate_relative_bandpower(self, psd_df: pd.DataFrame, normalize_to="bands"):
        """
        Calculate relative bandpower from PSD DataFrame.

        :param psd_df: PSD DataFrame with columns 'Frequency_Hz' and 'PSD_V2_per_Hz'
        :param normalize_to: 'total' → normalize to total PSD power (default);
                            'bands' → normalize only to sum of power in specified bands
        :return: dict of relative power values per band
        """

        io_stuff = self.io_instance
        # column names
        freq_col = io_stuff.psd_freq_col
        power_col = io_stuff.psd_power_col

        freqs = psd_df[freq_col].values
        power = psd_df[power_col].values

        # Compute total power for denominator depending on strategy
        if normalize_to == "total":
            total_power = np.trapezoid(power, freqs)  # estimate total power (i.e. AUC) with trapezoid rule
        elif normalize_to == "bands":
            # Only sum the power within all band ranges
            mask = np.zeros_like(freqs, dtype=bool)
            for low, high in self.frequency_bands.values():
                mask |= (freqs >= low) & (freqs < high)  # bitwise OR to use all band values or zeros
            total_power = np.trapezoid(power[mask], freqs[mask])
        else:
            raise ValueError("normalize_to must be either 'total' or 'bands'")

        result = {}
        for band_name, (low, high) in self.frequency_bands.items():  # for each frequency band (specified in class)
            mask = (psd_df[freq_col] >= low) & (psd_df[freq_col] < high)  # gather frequencies of band
            band_power = np.trapezoid(psd_df.loc[mask, power_col], psd_df.loc[mask, freq_col])
            relative_power = band_power / total_power if total_power > 0 else 0
            result[band_name] = relative_power
        return result

    def extract_shannon_entropy_for_parameter_combination(self):
        """
        Iterates over all PSD CSVs in the Feature PSD folder, calculates Shannon entropy,
        and saves the results as CSV in a Shannon_entropy output folder of the current parameter combination.
        """
        loader = self.data_loader
        saver = self.result_saver
        # create path to directory with PSDs (specified by class attributes)
        psd_directory_path = loader.create_psd_path_with_parameters(self.parameter_dict)

        # retrieve name of feature -> will be the name for subdirectory and column
        entropy_name = saver.shannon_entropy_subdir

        all_rows = []

        for psd_file in os.listdir(psd_directory_path):
            if psd_file.endswith(".csv"):
                psd_df, start, end, result_id = loader.load_psd_with_start_end_resultid(psd_directory_path, psd_file)

                # Calculate entropy
                entropy = self.calculate_shannon_entropy(psd_df)

                # Create result row
                result_row = {
                    "Start": start,
                    "End": end,
                    "ResultID": result_id,
                    entropy_name: entropy
                }
                all_rows.append(result_row)

        saver.save_feature_summary_episode(all_rows, entropy_name, self.parameter_dict)
        print(f"Succesfully calculated and saved Shannon Entropy")

    def calculate_shannon_entropy(self, psd_df: pd.DataFrame, normalize=True) -> float:
        """
        Calculates the Shannon entropy of a PSD.

        :param normalize: Normalize Entropy to [0,1] (default: True)
        :param psd_df: DataFrame with a 'PSD_V2_per_Hz' column
        :return: Shannon entropy as float
        """

        # columns
        io_stuff = self.io_instance
        power_col = io_stuff.psd_power_col

        # get all powers and calculate total power
        power = psd_df[power_col].values
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

    def extraxt_spectral_skewness_for_parameter_combination(self):
        """
        Iterates over all PSD CSVs in the Feature PSD folder, calculates spectral Skewness,
        and saves the results as CSV in a Spectral_skewness output folder of the current parameter combination.
        """
        loader = self.data_loader
        saver = self.result_saver
        # create path to directory with PSDs (specified by class attributes)
        psd_directory_path = loader.create_psd_path_with_parameters(self.parameter_dict)

        # retrieve name of feature -> will be the name for subdirectory and column
        skewness_name = saver.spectral_skewness_subdir

        all_rows = []

        for psd_file in os.listdir(psd_directory_path):
            if psd_file.endswith(".csv"):
                psd_df, start, end, result_id = loader.load_psd_with_start_end_resultid(psd_directory_path, psd_file)

                # Calculate skewness
                skewness = self.calculate_spectral_skewness(psd_df)

                # Create result row
                result_row = {
                    "Start": start,
                    "End": end,
                    "ResultID": result_id,
                    skewness_name: skewness
                }
                all_rows.append(result_row)

        saver.save_feature_summary_episode(all_rows, skewness_name, self.parameter_dict)
        print(f"Succesfully calculated and saved Spectral Skewness")

    def calculate_spectral_skewness(self, psd_df: pd.DataFrame, normalize="tanh", to_0_1=True) -> float:
        """
        Calculates spectral skewness from a power spectral density (PSD).

        :param psd_df: DataFrame with columns 'Freq_Hz' and 'PSD_V2_per_Hz'
        :param normalize: False -> raw, "tanh" -> confining smoothly to [-1,1] , or "clip" -> clipped to [-1,1]
        :param to_0_1: Assign all values from [-1,1] to [0,1]
        :return skewness: Float (normalized if requested)
        """
        io_stuff = self.io_instance

        # get frequencies and power
        freqs = psd_df[io_stuff.psd_freq_col].values
        power = psd_df[io_stuff.psd_power_col].values
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
        elif normalize == "clip":
            skewness = np.clip(skewness, -1.0, 1.0)

        if to_0_1:
            skewness = (skewness + 1) / 2

        return skewness
