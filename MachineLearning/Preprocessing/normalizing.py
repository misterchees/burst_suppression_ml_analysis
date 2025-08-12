import numpy as np
from typing import Union
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from MachineLearning.IO.load_data import LoadData
from MachineLearning.IO.save_result import SaveResult

class Normalizing:

    def __init__(self, method: str):
        """
        Initializes the Normalizing class with a normalization method.
        :param method: The normalization method. Options: 'zscore', 'tanh_zscore', 'minmax', 'quantile', 'log', 'clip'.
        """
        if method not in ["zscore", "tanh_zscore", "minmax", "quantile", "log", "clip"]:
            raise ValueError(f"Unsupported normalization method '{method}'. "
                             f"Supported methods are: 'zscore', 'tanh_zscore', 'minmax', 'quantile', 'log', 'clip'.")
        self.method = method

    def normalize_multiple_eeg(self, eeg_list: list, filtered_eeg: bool = True):
        loader = LoadData()
        saver = SaveResult()

        for result_id in eeg_list:
            fs, eeg_track = loader.load_eeg_data(result_id, filtered_eeg)
            normalized_eeg = self.normalize_array(eeg_track)
            saver.save_eeg_track(normalized_eeg, fs, result_id, ["normalized_data"])

    def normalize_scalar(self, value: float, feature_type: str = None) -> float:
        """
        Normalizes a single scalar value according to a method and optional feature type.

        :param value: The scalar value to normalize.
        :param feature_type: Optional context about the feature type (e.g., 'skewness', 'entropy').
        :returns: The normalized scalar.
        """
        if self.method == "tanh":
            return np.tanh(value)

        if self.method == "log":
            if value < 0:
                raise ValueError("Log normalization requires non-negative values.")
            return np.log1p(value)

        if self.method == "clip":
            if feature_type == "skewness":
                return np.clip(value, -2, 2)
            elif feature_type == "kurtosis":
                return np.clip(value, 0, 10)
            else:
                raise ValueError(f"Clipping not defined for feature type '{feature_type}'.")

        raise ValueError(f"Unsupported normalization method for scalars: '{self.method}'."
                         f"Supported methods are: 'tanh', 'log', 'clip'. ")


    def normalize_array(self, values: Union[np.ndarray, list], feature_type: str = None) -> np.ndarray:
        """
        Normalizes an array of values based on the method given in the class and an optional feature type.

        :param values: Array of values to normalize.

        :param feature_type: Optional context for clipping or specialized rules.
        :returns: Normalized array (NumPy ndarray).
        """
        x = np.asarray(values)

        if self.method == "zscore":
            return (x - np.mean(x)) / np.std(x)

        if self.method == "tanh_zscore":
            z = (x - np.mean(x)) / np.std(x)
            return np.tanh(z)

        if self.method == "minmax":
            scaler = MinMaxScaler()
            return scaler.fit_transform(x.reshape(-1, 1)).flatten()

        if self.method == "quantile":
            qt = QuantileTransformer(output_distribution='uniform', random_state=42)
            return qt.fit_transform(x.reshape(-1, 1)).flatten()

        if self.method == "log":
            if np.any(x < 0):
                raise ValueError("Log normalization requires all values to be non-negative.")
            return np.log1p(x)

        if self.method == "clip":
            if feature_type == "skewness":
                return np.clip(x, -2, 2)
            elif feature_type == "kurtosis":
                return np.clip(x, 0, 10)
            else:
                raise ValueError(f"Clipping not defined for feature type '{feature_type}'.")

        raise ValueError(f"Somehow you managed to use an unsupported normalization method '{self.method}' "
                         f"for array normalization. This should be dealt with in the initialization method ;)")


    @staticmethod
    def deviation_from_center_scaled(x, a, b):
        """
        Normalize the absolute deviation of x from the midpoint of [a, b] to [0, 1].

        :param x: scalar or array
        :param a: lower bound of input range
        :param b: higher bound of input range
        :return: normalized deviation of x from midpoint of [a, b] -> 0 = center, 1 = edge
        """
        midpoint = (a + b) / 2
        return 2 * np.abs(x - midpoint) / (b - a)