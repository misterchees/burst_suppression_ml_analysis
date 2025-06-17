import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import freqz
from MachineLearning.Utils.filter_utils import FilterUtils

matplotlib.use('TkAgg')


class Plots:
    @staticmethod
    def plot_butterworth_filtering(fig_and_ax=None, fs=128, lowcut=0.5, highcut=30.0, order=4):
        """
        Takes butterworth filter input parameter and returns a plot, that can be shown.
        :param fig_and_ax: A Tuple with the figure and axes objects. For plotting in subplot grids
        :param fs: sampling frequency
        :param lowcut: lowcut frequency of butterworth filter
        :param highcut: highcut frequency of butterworth filter
        :param order: order of butterworth filter -> steepness of transition to filtered frequencies
        :return: Figure and Axes of plot (if show=True)
        """

        # create butterworth filter with parameters
        b, a = FilterUtils.design_butterworth(fs=fs, lowcut=lowcut, highcut=highcut, order=order)

        # compute bandpass; worN is for resolution. Higher values make smoother plots
        w, h = freqz(b, a, worN=8000)  # w in rad/sample

        # Convert Frequency in Hz
        frequencies = w * fs / (2 * np.pi)

        # Avoid abs(h) being 0 to avoid log(abs(h)) -> infinity
        magnitude = np.abs(h)
        magnitude = np.maximum(magnitude, 1e-10)  # Values can't get below 1e-10

        # Plot
        if fig_and_ax is not None:
            fig, ax = fig_and_ax
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(frequencies, 20 * np.log10(magnitude), 'b', label=f'Bandpass (order={order})')
        ax.set_title('Bandpass of Butterworth-Filter')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Gain [dB]')
        ax.grid(True)
        ax.set_xlim(0, fs / 2)
        ax.axvline(lowcut, color='red', linestyle='--', label=f'{lowcut} Hz')
        ax.axvline(highcut, color='red', linestyle='--', label=f'{highcut} Hz')
        ax.legend()
        fig.tight_layout()

        return fig, ax

    @staticmethod
    def plot_psd(fig_and_ax, freqs, power, label, color="blue", title="Power Spectral Density", log_scale=False):
        """
        Plots a Power Spectral Density (PSD).
        :param fig_and_ax: A Tuple with the figure and axes objects. For plotting in subplot grids
        :param freqs: (array-like) frequencies in Hz
        :param power: (array-like) Power of frequencies in V²
        :param label: (str) label of the plot
        :param color: (str) color of the plot
        :param title: title of the plot
        :param log_scale: If True, then power axis will be logarithmic
        :return: Figure and Axes of plot (if show=True)
        """
        if fig_and_ax is not None:
            fig, ax = fig_and_ax
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
        if log_scale:
            ax.semilogy(freqs, power, label=label, color=color)
        else:
            ax.plot(freqs, power, label=label, color=color)
        ax.set_xlabel("Frequenz [Hz]")
        ax.set_ylabel("Power [V²/Hz]" if not log_scale else "log(Power[V²/Hz])")
        ax.set_title(title)
        ax.grid(True)
        ax.set_xlim([min(freqs), max(freqs)])
        ax.legend()

        return fig, ax

    @staticmethod
    def create_subplot_grid(n_plots, cols=2, figsize=(12, 6)):
        """
        Create a grid of subplots for multiple plots.

        :param n_plots: Number of subplots needed.
        :param cols: Number of columns in the grid (default is 2).
        :param figsize: Tuple for the figure size (width, height).
        :returns: (fig, axes) tuple where axes is a flat list of Axes objects.
        """
        rows = (n_plots + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=figsize)
        axs = axs.flatten() if n_plots > 1 else [axs]
        return fig, axs[:n_plots]

    @staticmethod
    def align_axis(preserve_ax, rescale_ax, scale_x=False, scale_y=True, min_max_scale=False):
        """
        Takes two axes and aligns their chosen scales.
        :param preserve_ax: Ax of which the scale will be preserved. It's the template for the rescale_ax scale.
        :param rescale_ax: Ax of which the scale will be rescaled according to preserve_ax scale.
        :param scale_x: If True, x_scales will be rescaled.
        :param scale_y: If True, y_scales will be rescaled.
        :param min_max_scale: Instead of copying the scale of preserve_ax, the global minima and maxima of the scales
        of both axes will be applied to both axes.
        :return: Both (rescaled) axes.
        """
        if scale_y:
            if min_max_scale:
                ymin = min(min(preserve_ax.get_ylim()), min(rescale_ax.get_ylim()))
                ymax = max(max(preserve_ax.get_ylim()), max(rescale_ax.get_ylim()))
                # Set Global maxima and minima for both y-axis
                preserve_ax.set_ylim(ymin, ymax)
                rescale_ax.set_ylim(ymin, ymax)
            else:
                rescale_ax.set_ylim(preserve_ax.get_ylim())
        if scale_x:
            if min_max_scale:
                xmin = min(min(preserve_ax.get_xlim()), min(rescale_ax.get_xlim()))
                xmax = max(max(preserve_ax.get_xlim()), max(rescale_ax.get_xlim()))
                # Set Global maxima and minima for both y-axis
                preserve_ax.set_xlim(xmin, xmax)
                rescale_ax.set_xlim(xmin, xmax)
            else:
                rescale_ax.set_xlim(preserve_ax.get_xlim())

        return preserve_ax, rescale_ax

    @staticmethod
    def plot_roc_auc(X_test, y_true, svm_model):

        from sklearn.metrics import roc_curve, auc
        # y_true: deine Labels, y_score: decision_function oder predict_proba[:, 1]
        y_score = svm_model.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # Plot
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonale
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
