import numpy as np
import scipy.signal as signal
from typing import Tuple


class FilterUtils:
    @staticmethod
    def design_butterworth(fs: int, lowcut: float, highcut: float, order: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Designs a butterworth band filter and returns (b, a) tuple.
        :param fs: Sampling frequency of the signal, that will be filtered.
        :param lowcut: Lower bound of the bandpass filter (Hz)
        :param highcut: Higher bound of the bandpass filter (Hz)
        :param order: Order of the butterworth filter
        :return: (b, a) i.e. Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        """
        # Design Butterworth bandpass filter
        ny_freq = 0.5 * fs
        # Normed cuts on nyquist, because this is a digital filter
        low = lowcut / ny_freq
        high = highcut / ny_freq
        b, a = signal.butter(N=order, Wn=[low, high], btype='band')
        return b, a
