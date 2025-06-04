import numpy as np
from typing import Union
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer


class MathUtils:
    @staticmethod
    def rescale_range(t, a, b, c, d):
        """
        Maps a number t with linear interpolation from range [a,b] to range [c,d].
        :param t: number to rescale
        :param a: lower bound of original range
        :param b: higher bound of original range
        :param c: lower bound of new range
        :param d: higher bound of new range
        :return: rescaled number f(t) from this function
        """

        # No inverted ranges, no empty ranges
        assert b > a
        assert d > c

        # If input range = output range, then no transformation is needed
        if a == c and b == d:
            return t

        # point in original range
        step = (d - c) / (b - a)
        return c + (step * (t - a))

    @staticmethod
    def scaled_tanh(x, center=0.0, sensitivity=1.0, out_min=-1.0, out_max=1.0):
        """
        Apply tanh scaling to input x, centered and scaled by sensitivity,
        then map to [out_min, out_max].

        :param x: scalar or array
        :param center: value to center x around linearly (e.g., 3 for kurtosis)
        :param sensitivity: divisor for scaling (higher = flatter tanh)
        :param out_min: lower bound of output range
        :param out_max: upper bound of output range
        :return: Scaled and bounded output in [out_min, out_max]
        """
        z = (x - center) / sensitivity
        tanh_val = np.tanh(z)  # tanh confines every value into [-1,1] with tan hyperbolic
        return MathUtils.rescale_range(t=tanh_val, a=-1, b=1, c=out_min, d=out_max)

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

    @staticmethod
    def normalize_scalar(value: float, method: str, feature_type: str = None) -> float:
        """
        Normalizes a single scalar value according to a method and optional feature type.

        :param value: The scalar value to normalize.
        :param method: The normalization method. Options: 'tanh', 'log', 'clip'.
        :param feature_type: Optional context about the feature type (e.g., 'skewness', 'entropy').
        :returns: The normalized scalar.
        """
        if method == "tanh":
            return np.tanh(value)

        if method == "log":
            if value < 0:
                raise ValueError("Log normalization requires non-negative values.")
            return np.log1p(value)

        if method == "clip":
            if feature_type == "skewness":
                return np.clip(value, -2, 2)
            elif feature_type == "kurtosis":
                return np.clip(value, 0, 10)
            else:
                raise ValueError(f"Clipping not defined for feature type '{feature_type}'.")

        raise ValueError(f"Unsupported method '{method}' for scalar normalization.")

    @staticmethod
    def normalize_array(values: Union[np.ndarray, list], method: str, feature_type: str = None) -> np.ndarray:
        """
        Normalizes an array of values based on a method and optional feature type.

        :param values: Array of values to normalize.
        :param method: The normalization method. Options: 'zscore', 'tanh_zscore', 'minmax', 'quantile', 'log', 'clip'.
        :param feature_type: Optional context for clipping or specialized rules.
        :returns: Normalized array (NumPy ndarray).
        """
        x = np.asarray(values)

        if method == "zscore":
            return (x - np.mean(x)) / np.std(x)

        if method == "tanh_zscore":
            z = (x - np.mean(x)) / np.std(x)
            return np.tanh(z)

        if method == "minmax":
            scaler = MinMaxScaler()
            return scaler.fit_transform(x.reshape(-1, 1)).flatten()

        if method == "quantile":
            qt = QuantileTransformer(output_distribution='uniform', random_state=42)
            return qt.fit_transform(x.reshape(-1, 1)).flatten()

        if method == "log":
            if np.any(x < 0):
                raise ValueError("Log normalization requires all values to be non-negative.")
            return np.log1p(x)

        if method == "clip":
            if feature_type == "skewness":
                return np.clip(x, -2, 2)
            elif feature_type == "kurtosis":
                return np.clip(x, 0, 10)
            else:
                raise ValueError(f"Clipping not defined for feature type '{feature_type}'.")

        raise ValueError(f"Unsupported method '{method}' for array normalization.")
