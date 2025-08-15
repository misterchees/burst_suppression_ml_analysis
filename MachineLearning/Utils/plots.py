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
    def plot_psd(fig_and_ax, freqs, power, label, color="blue", title="Power Spectral Density",
                 log_scale=False, spread=None, alpha=0.3):
        """
        Plots a PSD with optional uncertainty shading.
        :param fig_and_ax: Tuple with (fig, ax) for subplot integration.
        :param freqs: Frequencies in Hz.
        :param power: Mean PSD values.
        :param label: Label for the line.
        :param color: Line color.
        :param title: Plot title.
        :param log_scale: If True, y-axis is logarithmic.
        :param spread: Optional uncertainty array (same shape as power).
        :param alpha: Transparency for shaded area.
        """
        if fig_and_ax is not None:
            fig, ax = fig_and_ax
        else:
            fig, ax = plt.subplots(figsize=(10, 5))

        if log_scale:
            ax.semilogy(freqs, power, label=label, color=color)
        else:
            ax.plot(freqs, power, label=label, color=color)

        if spread is not None:
            ax.fill_between(freqs, power - spread, power + spread, color=color, alpha=alpha)

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

    @staticmethod
    def plot_eeg_signal(eeg_signal: np.ndarray, fs: int, filtered: bool):
        """
        Plots a raw EEG signal with a time axis in seconds.

        :param eeg_signal: 1D NumPy array with EEG samples
        :param fs: Sampling frequency in Hz (default 128 Hz)
        :param filtered: If True, the plotted EEG signal is filtered, else raw. Influences chart name.
        """
        # time in seconds
        time = np.arange(len(eeg_signal)) / fs

        plt.figure(figsize=(15, 5))
        plt.plot(time, eeg_signal, linewidth=0.8)
        plt.xlabel("Time [s]")
        plt.ylabel("EEG Signal")
        plt.title(f"EEG {"filtered" if filtered else "raw"} Signal")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_two_channel_eeg(eeg_data: np.ndarray, fs: int, filtered: bool, channel_names=("Channel 1", "Channel 2")):
        """
        Plots a 2-channel EEG signal over time, with each channel in its own subplot.

        :param eeg_data: 2D NumPy array of shape (n_samples, 2)
        :param fs: Sampling frequency in Hz
        :param channel_names: Tuple with names for each channel
        :param filtered: If True, the plotted EEG signal is assumed to be filtered, else raw. Influences chart name.
        """
        if eeg_data.shape[1] != 2:
            raise ValueError("Input EEG data must have exactly 2 channels (shape: [n_samples, 2])")

        time = np.arange(eeg_data.shape[0]) / fs

        fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
        for i in range(2):
            axes[i].plot(time, eeg_data[:, i], linewidth=0.8)
            axes[i].set_ylabel(f"{channel_names[i]}")
            axes[i].grid(True)

        axes[1].set_xlabel("Time [s]")
        plt.suptitle(f"2-Channel EEG Signal - {"filtered" if filtered else "raw"}", fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Room for title
        plt.show()

    @staticmethod
    def plot_kmeans_results(X, labels, centroids=None, title="K-Means Clustering"):
        """
        Visualizes K-Means clustering results in 2D.

        :param X: Data array of shape (n_samples, 2). Must be 2D for plotting.
        :param labels: Array of cluster labels (length n_samples).
        :param centroids: Optional array of cluster centers of shape (n_clusters, 2).
        :param title: Title for the plot.
        :returns: None
        """
        if X.shape[1] != 2:
            raise ValueError("This plotting function only works for 2D data.")

        # Ensure numpy arrays
        X = np.array(X)
        labels = np.array(labels)

        # Create scatter plot
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", alpha=0.7, edgecolor="k")

        # Plot centroids if provided
        if centroids is not None:
            centroids = np.array(centroids)
            plt.scatter(centroids[:, 0], centroids[:, 1],
                        marker="*", s=300, c="red", edgecolor="k", label="Centroids")

        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.colorbar(scatter, label="Cluster")
        plt.show()

    @staticmethod
    def plot_components_2d(pca_result, title: str, labels=None, figsize=(8, 6),
                           jitter=False, jitter_strength=0.01, alpha=0.6,
                           marker_size=20, separate_plots=False):
        if pca_result is None:
            raise ValueError("Given PCA result is None.")

        X = pca_result[:, :2]  # First 2 PCs

        if jitter:
            noise = np.random.normal(0, jitter_strength, X.shape)
            X = X + noise

        figs_axes = []

        if labels is not None:
            labels = np.array(labels)
            unique_labels = np.unique(labels)

            if separate_plots:
                for ul in unique_labels:
                    mask = labels == ul
                    fig, ax = plt.subplots(figsize=figsize)
                    ax.scatter(X[mask, 0], X[mask, 1], alpha=alpha, s=marker_size, label=str(ul))
                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2")
                    ax.set_title(f"{title} - Label {ul}")
                    ax.legend()
                    ax.grid(True)
                    plt.tight_layout()
                    plt.show()
                    figs_axes.append((fig, ax))
            else:
                fig, ax = plt.subplots(figsize=figsize)
                for ul in unique_labels:
                    mask = labels == ul
                    ax.scatter(X[mask, 0], X[mask, 1], alpha=alpha, s=marker_size, label=str(ul))
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_title(title)
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                plt.show()
                figs_axes.append((fig, ax))
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax.scatter(X[:, 0], X[:, 1], alpha=alpha, s=marker_size)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(title)
            ax.grid(True)
            plt.tight_layout()
            plt.show()
            figs_axes.append((fig, ax))

        return figs_axes if separate_plots else figs_axes[0]

    @staticmethod
    def plot_components_3d(pca_result, title: str, labels=None, jitter=False,
                           jitter_strength=0.01, alpha=0.6, marker_size=20,
                           separate_plots=False):
        if pca_result is None:
            raise ValueError("Given PCA result is None.")

        X = pca_result[:, :3]  # First 3 PCs

        if jitter:
            noise = np.random.normal(0, jitter_strength, X.shape)
            X = X + noise

        figs_axes = []

        if labels is not None:
            labels = np.array(labels)
            unique_labels = np.unique(labels)

            if separate_plots:
                for ul in unique_labels:
                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111, projection='3d')
                    idx = labels == ul
                    ax.scatter(X[idx, 0], X[idx, 1], X[idx, 2], label=str(ul), alpha=alpha, s=marker_size)
                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2")
                    ax.set_zlabel("PC3")
                    ax.set_title(f"{title} - Label {ul}")
                    ax.legend()
                    plt.tight_layout()
                    plt.show()
                    figs_axes.append((fig, ax))
            else:
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                for ul in unique_labels:
                    idx = labels == ul
                    ax.scatter(X[idx, 0], X[idx, 1], X[idx, 2], label=str(ul), alpha=alpha, s=marker_size)
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_zlabel("PC3")
                ax.set_title(title)
                ax.legend()
                plt.tight_layout()
                plt.show()
                figs_axes.append((fig, ax))
        else:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=alpha, s=marker_size)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
            ax.set_title(title)
            plt.tight_layout()
            plt.show()
            figs_axes.append((fig, ax))

        return figs_axes if separate_plots else figs_axes[0]

    @staticmethod
    def plot_scree(variance_ratio, title="Scree_plot"):
        if variance_ratio is None:
            raise ValueError("Given Variance Ratio is None.")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(np.arange(1, len(variance_ratio) + 1), variance_ratio, marker='o')
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_title(title)
        ax.grid(True)
        plt.tight_layout()
        plt.show()
        return fig, ax
