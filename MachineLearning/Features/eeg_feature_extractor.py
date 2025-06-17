import os
from nolds import sampen
from antropy import perm_entropy
from EntropyHub import FuzzEn
import pandas as pd
import numpy as np
from numpy import floating

from MachineLearning.Core.ml_object import MLObject
from MachineLearning.IO.save_result import SaveResult
from MachineLearning.IO.load_data import LoadData, load_psd_with_start_end_resultid
from MachineLearning.Utils.math_utils import MathUtils
from MachineLearning.Utils.feature_utils import FeatureUtils
from MachineLearning.Features.feature_function import FeatureFunction

# Registries for feature functions. One for the calculators and one for the extractors
feature_calculators_registry = {}
feature_extractors_registry = {}


def register_feature_calculator(name):
    """Adds feature calculator function names to registry. Callable by @register_feature(name) before function code

    :param name: Key of the feature. To find in path config.
    """

    def decorator(func):
        feature_calculators_registry[name] = func
        return func

    return decorator


def register_feature_extractor(name, default_params=None):
    """
    Decorator to register a feature function as a FeatureFunction object.

    :param name: Feature key for the registry.
    :param default_params: Optional default parameters for this feature.
    :returns: Decorator that registers the function.
    """
    def decorator(func):
        feature_func = FeatureFunction(func, default_params)
        feature_extractors_registry[name] = feature_func
        return func  # return func to keep it callable
    return decorator


class EEGFeatureExtractor(MLObject):
    data_loader = LoadData()
    result_saver = SaveResult()

    def __init__(self, faw: bool, awake: bool, normal_an: bool):
        """
        Create a FeatureExtractor object.
        :param faw: Boolean value to indicate if instance currently handles fake awake (True) or awake (False) data.
        :param awake: Boolean value to indicate if instance currently handles awake data.
        """
        super().__init__(faw, awake, normal_an)
        # initialize registries
        self.feature_calc_funcs = feature_calculators_registry
        self.feature_extract_funcs = feature_extractors_registry

    @register_feature_extractor("bandpower", default_params={"normalize_to": "bands"})
    def extract_relative_bandpower(self, normalize_to="bands"):
        """
        Iterates over all PSD CSVs in the Feature PSD folder,calculates the relative band power
        and saves the results as CSV in a Rel_bandpower output folder of the current parameter combination.
        :param normalize_to: 'total' → normalize to total PSD power (default);
                            'bands' → normalize only to sum of power in specified bands
        """

        # retrieve name of feature -> will be the name for subdirectory and column
        feature_key = "bandpower"

        # Extract and save feature
        self.extract_feature_from_PSDs(feature_key, normalize_to=normalize_to)

    @register_feature_calculator("bandpower")
    def calculate_relative_bandpower(self, psd_df: pd.DataFrame, normalize_to="bands") -> dict:
        """
        Calculate relative bandpower from PSD DataFrame.

        :param psd_df: PSD DataFrame with columns 'Frequency_Hz' and 'PSD_V2_per_Hz'
        :param normalize_to: 'total' → normalize to total PSD power (default);
                            'bands' → normalize only to sum of power in specified bands
        :return: dict of relative power values per band
        """

        loader = self.data_loader
        frequency_bands = self.param_config["frequency_bands"]
        # column names
        psd_cols = loader.data_names["psd_files"]
        freq_col = psd_cols["psd_freq_col"]
        power_col = psd_cols["psd_power_col"]

        freqs = psd_df[freq_col].values
        power = psd_df[power_col].values

        # Compute total power for denominator depending on strategy
        if normalize_to == "total":
            total_power = np.trapezoid(power, freqs)  # estimate total power (i.e. AUC) with trapezoid rule
        elif normalize_to == "bands":
            # Only sum the power within all band ranges
            mask = np.zeros_like(freqs, dtype=bool)
            for low, high in frequency_bands.values():
                mask |= (freqs >= low) & (freqs < high)  # bitwise OR to use all band values or zeros
            total_power = np.trapezoid(power[mask], freqs[mask])
        else:
            raise ValueError("normalize_to must be either 'total' or 'bands'")

        result = {}
        for band_name, (low, high) in frequency_bands.items():  # for each frequency band (specified in class)
            mask = (psd_df[freq_col] >= low) & (psd_df[freq_col] < high)  # gather frequencies of band
            band_power = np.trapezoid(psd_df.loc[mask, power_col], psd_df.loc[mask, freq_col])
            relative_power = band_power / total_power if total_power > 0 else 0
            result[band_name] = relative_power
        return result

    @register_feature_extractor("shannon_entropy", default_params={"normalize": True})
    def extract_shannon_entropy(self, normalize=True):
        """
        Iterates over all PSD CSVs in the Feature PSD folder, calculates Shannon entropy,
        and saves the results as CSV in a Shannon_entropy output folder of the current parameter combination.

        :param normalize: Normalize Entropy to [0,1] (default: True)
        """

        # retrieve name of feature -> will be the name for subdirectory and column
        feature_key = "shannon_entropy"

        # Extract and save feature
        self.extract_feature_from_PSDs(feature_key, normalize=normalize)

    @register_feature_calculator("shannon_entropy")
    def calculate_shannon_entropy(self, psd_df: pd.DataFrame, normalize=True) -> float:
        """
        Calculates the Shannon entropy of a PSD.

        :param normalize: Normalize Entropy to [0,1] (default: True)
        :param psd_df: DataFrame with a 'PSD_V2_per_Hz' column
        :return: Shannon entropy as float
        """

        # columns
        power_col = self.data_loader.data_names["psd_files"]["psd_power_col"]

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

    @register_feature_extractor("spectral_skewness", default_params={"normalize": True, "n_method": "clip",
                                                                     "lower_bound": 0, "upper_bound": 1})
    def extraxt_spectral_skewness(self, normalize=True, n_method="clip", lower_bound=0, upper_bound=1):
        """
        Iterates over all PSD CSVs in the Feature PSD folder, calculates spectral Skewness,
        and saves the results as CSV in a Spectral_skewness output folder of the current parameter combination.

        :param normalize: False -> raw; True -> normalized with method specified in n_method
        :param n_method: "tanh" -> confines values with tan hyperbolic into [a,b];
                         "clip" -> takes only values between [a,b], outliers get mapped respectively to a or b
        :param lower_bound: Lower bound of normalization range, referred to as a
        :param upper_bound: Upper bound of normalization range, referred to as b
        """
        # retrieve name of feature -> will be the name for subdirectory and column
        feature_key = "spectral_skewness"

        # Extract and save feature
        self.extract_feature_from_PSDs(feature_key, normalize=normalize, n_method=n_method,
                                       lower_bound=lower_bound, upper_bound=upper_bound)

    @register_feature_calculator("spectral_skewness")
    def calculate_spectral_skewness(self, psd_df: pd.DataFrame, normalize=True, n_method="clip",
                                    lower_bound=0, upper_bound=1) -> float:
        """
        Calculates spectral skewness from a power spectral density (PSD).

        :param psd_df: DataFrame with columns 'Freq_Hz' and 'PSD_V2_per_Hz'
        :param normalize: False -> raw, True -> normalized with method specified in n_method
        :param n_method: "tanh" -> confines values with tan hyperbolic into [a,b];
                         "clip" -> takes only values between [a,b], outliers get mapped respectively to a or b
        :param lower_bound: Lower bound of normalization range, referred to as a
        :param upper_bound: Upper bound of normalization range, referred to as b
        :return: Skewness as float (normalized if requested)
        """
        io_stuff = self.data_loader

        # get frequencies and power
        psd_cols = io_stuff.data_names["psd_files"]
        freqs = psd_df[psd_cols["psd_freq_col"]].values
        power = psd_df[psd_cols["psd_power_col"]].values
        total_power = np.sum(power)

        if total_power == 0:
            return np.nan

        # Normalize power to form a probability distribution
        probabilities = power / total_power
        # Weighted mean frequency
        mean_freq = np.sum(probabilities * freqs)
        # Weighted standard deviation
        std_freq = np.sqrt(np.sum(probabilities * (freqs - mean_freq) ** 2))
        # compute spectral skewness
        skewness = np.sum(probabilities * ((freqs - mean_freq) / std_freq) ** 3)

        if normalize:
            if n_method == "tanh":
                skewness = MathUtils.scaled_tanh(skewness, out_min=lower_bound, out_max=upper_bound)
            elif n_method == "clip":
                skewness = np.clip(skewness, lower_bound, upper_bound)
        else:
            raise ValueError(f"Normalization method '{n_method}' not supported")

        return skewness

    @register_feature_extractor("spectral_kurtosis", default_params={"normalize": True, "n_method": "clip",
                                                                     "lower_bound": 0, "upper_bound": 1})
    def extraxt_spectral_kurtosis(self, normalize=True, n_method="clip", lower_bound=0, upper_bound=1):
        """
        Iterates over all PSD CSVs in the Feature PSD folder, calculates spectral Kurtosis,
        and saves the results as CSV in a Spectral_kurtosis output folder of the current parameter combination.

        :param normalize: False -> raw; True -> normalized with method specified in n_method
        :param n_method: "tanh" -> confines values with tan hyperbolic into [a,b];
                         "clip" -> takes only values between [a,b], outliers get mapped respectively to a or b
        :param lower_bound: Lower bound of normalization range, referred to as a
        :param upper_bound: Upper bound of normalization range, referred to as b
        :return: Spectral kurtosis as float
        """
        # retrieve name of feature -> will be the name for subdirectory and column
        feature_key = "spectral_kurtosis"

        # Calculate results for PSDs
        self.extract_feature_from_PSDs(feature_key, normalize=normalize, n_method=n_method,
                                       lower_bound=lower_bound, upper_bound=upper_bound)

    @register_feature_calculator("spectral_kurtosis")
    def calculate_spectral_kurtosis(self, psd_df: pd.DataFrame, normalize=True, n_method="clip",
                                    lower_bound=0, upper_bound=1) -> float:
        """
        Calculates spectral kurtosis from a power spectral density (PSD).

        :param psd_df: DataFrame containing the PSD with frequency and power columns
        :param normalize: False -> raw, True -> normalized with method specified in n_method
        :param n_method: "tanh" -> confines values with tan hyperbolic into [a,b];
                         "clip" -> takes only values between [a,b], outliers get mapped respectively to a or b
        :param lower_bound: Lower bound of normalization range, referred to as a
        :param upper_bound: Upper bound of normalization range, referred to as b
        :return: Spectral kurtosis as float (normalized if requested)
        """
        io_stuff = self.data_loader

        # get frequencies and power
        psd_cols = io_stuff.data_names["psd_files"]
        freqs = psd_df[psd_cols["psd_freq_col"]].values
        power = psd_df[psd_cols["psd_power_col"]].values
        total_power = np.sum(power)

        if total_power == 0:
            return np.nan

        # Normalize power to probabilities (weights)
        probabilities = power / total_power
        mean_freq = np.sum(probabilities * freqs)
        std_freq = np.sqrt(np.sum(probabilities * (freqs - mean_freq) ** 2))

        # Weighted fourth central moment
        kurtosis = np.sum(probabilities * ((freqs - mean_freq) / std_freq) ** 4)

        if normalize:
            if n_method == "tanh":
                kurtosis = MathUtils.scaled_tanh(kurtosis, out_min=lower_bound, out_max=upper_bound)
            elif n_method == "clip":
                kurtosis = np.clip(kurtosis, lower_bound, upper_bound)
        else:
            raise ValueError(f"Normalization method '{n_method}' not supported")

        return kurtosis

    @register_feature_extractor("mean", default_params={"channel": 1})
    def extract_mean(self, channel=1):
        """
        Calculates the mean value of EEG signal for given channel of each episode from the subdirectory
        specified by current parameters.

        :param channel: channel of the EEG; valid options -> 1 or 2
        """

        # Get all Episodes (default = filtered episodes)
        self.update_current_epochs(channel)

        # retrieve name of feature -> will be the name for subdirectory and column
        feature_key = "mean"

        # Calculate and save results for epochs
        self.extract_feature_from_EEG(feature_key)


    @register_feature_calculator("mean")
    def calculate_mean(self, signal: np.ndarray) -> floating:
        """
        Calculates the mean value of EEG signal.

        :param signal: EEG signal to be averaged.
        :return: Mean value of EEG signal
        """
        return np.mean(signal)

    @register_feature_extractor("variance", default_params={"channel": 1})
    def extract_variance(self, channel=1):
        """
        Calculates the variance of EEG signal for given channel of each episode from the subdirectory
        specified by current parameters.

        :param channel: channel of the EEG; valid options -> 1 or 2
        """

        # Get all Episodes (default= filtered episodes)
        self.update_current_epochs(channel)

        # retrieve name of feature -> will be the name for subdirectory and column
        feature_key = "variance"

        # Calculate and save results for epochs
        self.extract_feature_from_EEG(feature_key)

    @register_feature_calculator("variance")
    def calculate_variance(self, signal: np.ndarray) -> floating:
        """
        Computes the variance of the EEG signal.

        :param signal: EEG signal.
        :returns: Variance of the signal.
        """
        return np.var(signal)

    @register_feature_extractor("amplitude", default_params={"channel": 1})
    def extract_amplitude(self, channel=1):
        """
        Calculates the peak-to-peak Amplitude of EEG signal for given channel of each episode from the subdirectory
        specified by current parameters.

        :param channel: channel of the EEG; valid options -> 1 or 2
        """
        # Get all Episodes (default= filtered episodes)
        self.update_current_epochs(channel)

        # retrieve name of feature -> will be the name for subdirectory and column
        feature_key = "amplitude"

        # Calculate and save results for epochs
        self.extract_feature_from_EEG(feature_key)

    @register_feature_calculator("amplitude")
    def calculate_amplitude(self, signal: np.ndarray) -> float:
        """
        Computes the peak-to-peak amplitude of the EEG signal.

        :param signal: EEG signal.
        :returns: Peak-to-peak amplitude.
        """
        return np.ptp(signal)  # equivalent to max - min

    @register_feature_extractor("sample_entropy", default_params={"channel": 1, "emb_dim": 2, "tolerance": 0.2})
    def extract_sample_entropy(self, channel=1, emb_dim: int = 2, tolerance: float = 0.2):
        """
        Calculates the sample entropy of EEG signal for given channel of each episode from the subdirectory
        specified by current parameters.

        :param channel: channel of the EEG; valid options -> 1 or 2
        :param emb_dim: Embedding dimension (default 2).
        :param tolerance: Tolerance as a fraction of std (default 0.2).
        """
        # Get all Episodes (default= filtered episodes)
        self.update_current_epochs(channel)

        # retrieve name of feature -> will be the name for subdirectory and column
        feature_key = "sample_entropy"

        # Calculate and save results for epochs
        self.extract_feature_from_EEG(feature_key, emb_dim=emb_dim, tolerance=tolerance)

    @register_feature_calculator("sample_entropy")
    def calculate_sample_entropy(self, signal: np.ndarray, emb_dim: int = 2, tolerance: float = 0.2) -> float:
        """
        Computes the sample entropy of the EEG signal.

        :param signal: 1D EEG signal.
        :param emb_dim: Embedding dimension (default 2).
        :param tolerance: Tolerance as a fraction of std (default 0.2).
        :returns: Sample entropy value.
        """
        return sampen(signal, emb_dim=emb_dim, tolerance=tolerance * np.std(signal))

    @register_feature_extractor("permutation_entropy", default_params={"channel": 1, "order": 3,
                                                                       "delay": 1, "normalize": True})
    def extract_permutation_entropy(self, channel=1, order: int = 3, delay: int = 1, normalize: bool = True):
        """
        Calculates the permutation entropy of EEG signal for given channel of each episode from the subdirectory
        specified by current parameters.

        :param channel: channel of the EEG; valid options -> 1 or 2
        :param order: Embedding order (default 3).
        :param delay: Delay between elements in embedded vectors (default 1).
        :param normalize: Whether to normalize the entropy (default True).
        """
        # Get all Episodes (default= filtered episodes)
        self.update_current_epochs(channel)

        # retrieve name of feature -> will be the name for subdirectory and column
        feature_key = "permutation_entropy"

        # Calculate and save results for epochs
        self.extract_feature_from_EEG(feature_key, order=order, delay=delay, normalize=normalize)

    @register_feature_calculator("permutation_entropy")
    def calculate_permutation_entropy(self, signal: np.ndarray, order: int = 3, delay: int = 1,
                                      normalize: bool = True) -> float:
        """
        Computes the permutation entropy of the EEG signal.

        :param signal: EEG signal.
        :param order: Embedding order (default 3).
        :param delay: Delay between elements in embedded vectors (default 1).
        :param normalize: Whether to normalize the entropy (default True).
        :returns: Permutation entropy value.
        """
        return perm_entropy(signal, order, delay, normalize)

    @register_feature_extractor("fuzzy_entropy", default_params={"channel": 1, "m": 2, "r": 0.2, "n": 2})
    def extract_fuzzy_entropy(self, channel=1, m: int = 2, r: float = 0.2, n: int = 2):
        """
        Calculates the fuzzy entropy of EEG signal for given channel of each episode from the subdirectory
        specified by current parameters.

        :param channel: channel of the EEG; valid options -> 1 or 2
        :param m: Embedding dimension.
        :param r: Tolerance (relative to std).
        :param n: Fuzziness parameter.
        """
        # Get all Episodes (default= filtered episodes)
        self.update_current_epochs(channel)

        # retrieve name of feature -> will be the name for subdirectory and column
        feature_key = "fuzzy_entropy"

        # Calculate and save results for epochs
        self.extract_feature_from_EEG(feature_key, m=m, r=r, n=n)

    @register_feature_calculator("fuzzy_entropy")
    def calculate_fuzzy_entropy(self, signal: np.ndarray, m: int = 2, r: float = 0.2, n: int = 2) -> float:
        """
        Computes fuzzy entropy of EEG-signal.

        :param signal: 1D EEG signal.
        :param m: Embedding dimension.
        :param r: Tolerance (relative to std).
        :param n: Fuzziness parameter.
        :returns: Fuzzy entropy value.
        """
        std_r = r * np.std(signal)
        result = FuzzEn(signal, m, std_r, n)
        return result['FuzzEn']

    def extract_feature_from_EEG(self, feature_key: str, **kwargs):
        """
        Helper function to automate calculation of feature defined by feature key.

        :param feature_key: Key that defines directory, feature name in file and feature function applied
        :param kwargs: Optional additional parameters for feature function
        :return: A list of metadata und the feature itself
        """
        if feature_key not in self.feature_calc_funcs:
            raise ValueError(f"Feature '{feature_key}' not found.")

        # Load function for this feature
        func = self.feature_calc_funcs[feature_key]

        # get feature name
        feature_name = self.data_loader.return_feature_name(feature_key)

        faw_all_rows = []
        awake_all_rows = []
        if self.faw:
            for start, end, result_id, fs, eeg_segment in self.faw_epochs.epoch_times:
                value = func(self, eeg_segment, **kwargs)
                faw_all_rows.append({"Start": start, "End": end, "ResultID": result_id, feature_name: value})
            self.result_saver.save_faw_feature_summary_episode(faw_all_rows, feature_key, self.parameter_dict)
            print(f"Succesfully calculated and saved {feature_name} of fake awake EEG-signals")

        if self.awake:
            for start, end, result_id, fs, eeg_segment in self.awake_epochs.epoch_times:
                value = func(self, eeg_segment, **kwargs)
                awake_all_rows.append({"Start": start, "End": end, "ResultID": result_id, feature_name: value})
            self.result_saver.save_awake_feature_summary_episode(awake_all_rows, feature_key, self.parameter_dict)
            print(f"Succesfully calculated and saved {feature_name} of true awake EEG-signals")

        if self.normal_an:
            for start, end, result_id, fs, eeg_segment in self.normal_an_epochs.epoch_times:
                value = func(self, eeg_segment, **kwargs)
                awake_all_rows.append({"Start": start, "End": end, "ResultID": result_id, feature_name: value})
            self.result_saver.save_normal_an_feature_summary_episode(awake_all_rows, feature_key, self.parameter_dict)
            print(f"Succesfully calculated and saved {feature_name} of normal anesthesia awake EEG-signals")

    def extract_feature_from_PSDs(self, feature_key: str, **kwargs):

        if feature_key not in self.feature_calc_funcs:
            raise ValueError(f"Feature '{feature_key}' not found.")

        # Load function for this feature
        func = self.feature_calc_funcs[feature_key]

        # get feature name
        feature_name = self.data_loader.return_feature_name(feature_key)

        faw_all_rows = []
        awake_all_rows = []
        if self.faw:
            psd_directory_path = self.data_loader.psd_folder_path(self.parameter_dict)
            for psd_file in os.listdir(psd_directory_path):
                if psd_file.endswith(".csv"):
                    psd_df, start, end, result_id = load_psd_with_start_end_resultid(psd_directory_path, psd_file)

                    # Calculate feature with feature function
                    value = func(self, psd_df, **kwargs)
                    # For features that return multiple values (as a dict)
                    if isinstance(value, dict):
                        faw_all_rows.append({"Start": start, "End": end, "ResultID": result_id, **value})
                    else:
                        faw_all_rows.append({"Start": start, "End": end, "ResultID": result_id, feature_name: value})
            self.result_saver.save_faw_feature_summary_episode(faw_all_rows, feature_key, self.parameter_dict)
            print(f"Succesfully calculated and saved {feature_name} for fake awake PSDs")

        if self.awake:
            psd_directory_path = self.data_loader.psd_folder_path(self.parameter_dict, False)
            for psd_file in os.listdir(psd_directory_path):
                if psd_file.endswith(".csv"):
                    psd_df, start, end, result_id = load_psd_with_start_end_resultid(psd_directory_path, psd_file)

                    # Calculate feature with feature function
                    value = func(self, psd_df, **kwargs)
                    # For features that return multiple values (as a dict)
                    if isinstance(value, dict):
                        awake_all_rows.append({"Start": start, "End": end, "ResultID": result_id, **value})
                    else:
                        awake_all_rows.append({"Start": start, "End": end, "ResultID": result_id, feature_name: value})
            self.result_saver.save_awake_feature_summary_episode(awake_all_rows, feature_key, self.parameter_dict)
            print(f"Succesfully calculated and saved {feature_name} for awake PSDs")

        if self.normal_an:
            psd_directory_path = self.data_loader.psd_folder_path(self.parameter_dict, normal_an=True)
            for psd_file in os.listdir(psd_directory_path):
                if psd_file.endswith(".csv"):
                    psd_df, start, end, result_id = load_psd_with_start_end_resultid(psd_directory_path, psd_file)

                    # Calculate feature with feature function
                    value = func(self, psd_df, **kwargs)
                    # For features that return multiple values (as a dict)
                    if isinstance(value, dict):
                        awake_all_rows.append({"Start": start, "End": end, "ResultID": result_id, **value})
                    else:
                        awake_all_rows.append({"Start": start, "End": end, "ResultID": result_id, feature_name: value})
            self.result_saver.save_normal_an_feature_summary_episode(awake_all_rows, feature_key, self.parameter_dict)
            print(f"Succesfully calculated and saved {feature_name} for awake PSDs")

    def combine_all_features(self):
        if self.faw:
            FeatureUtils.combine_features(self.parameter_dict)
        if self.awake:
            FeatureUtils.combine_features(self.parameter_dict, faw=False)
        if self.normal_an:
            FeatureUtils.combine_features(self.parameter_dict, normal_an=True)

    def combine_features(self, *features):
        if self.faw:
            FeatureUtils.combine_features(self.parameter_dict, False, *features)
        if self.awake:
            FeatureUtils.combine_features(self.parameter_dict, False, False, False, *features)
        if self.normal_an:
            FeatureUtils.combine_features(self.parameter_dict, False, False, True, *features)
