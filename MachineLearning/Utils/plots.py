import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, freqz


class Plots:
    @staticmethod
    def plot_butterworth_filtering(fs=128, lowcut=0.5, highcut=30.0, order=4):
        """
        Takes butterworth filter input parameter and plots the filtering curve
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

        # compute bandpass
        w, h = freqz(b, a, worN=8000)  # w in rad/sample

        # Convert Frequency in Hz
        frequencies = w * fs / (2 * np.pi)

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(frequencies, 20 * np.log10(abs(h)), 'b')
        plt.title('Frequenzgang des Butterworth-Bandpassfilters')
        plt.xlabel('Frequenz [Hz]')
        plt.ylabel('Verstärkung [dB]')
        plt.grid(True)
        plt.xlim(0, fs / 2)
        plt.axvline(lowcut, color='red', linestyle='--', label=f'{lowcut} Hz')
        plt.axvline(highcut, color='red', linestyle='--', label=f'{highcut} Hz')
        plt.legend()
        plt.tight_layout()
        plt.show()
