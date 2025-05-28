import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import butter, freqz


class Plots:
    @staticmethod
    def plot_butterworth_filtering(fs=128, lowcut=0.5, highcut=30.0, order=4):
        """
        Takes butterworth filter input parameter and returns a plot, that can be shown.
        :param fs: sampling frequency
        :param lowcut: lowcut frequency of butterworth filter
        :param highcut: highcut frequency of butterworth filter
        :param order: order of butterworth filter -> steepness of transition to filtered frequencies
        """

        # Nyquist-normalized edge frequency
        nyq = fs / 2
        wn = [lowcut / nyq, highcut / nyq]

        # create butterworth filter with parameters
        b, a = butter(order, wn, btype='band')

        # compute bandpass; worN is for resolution. Hichger values make smoother plots
        w, h = freqz(b, a, worN=8000)  # w in rad/sample

        # Convert Frequency in Hz
        frequencies = w * fs / (2 * np.pi)

        # Avoid abs(h) being 0 to avoid log(abs(h)) -> infinity
        magnitude = np.abs(h)
        magnitude = np.maximum(magnitude, 1e-10)  # Values can't get below 1e-10

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(frequencies, 20 * np.log10(magnitude), 'b', label=f'Bandpass (order={order})')
        plt.title('Bandpass of Butterworth-Filter')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplification [dB]')
        plt.grid(True)
        plt.xlim(0, fs / 2)
        plt.axvline(lowcut, color='red', linestyle='--', label=f'{lowcut} Hz')
        plt.axvline(highcut, color='red', linestyle='--', label=f'{highcut} Hz')
        plt.legend()
        plt.tight_layout()

        return plt
