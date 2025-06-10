from matplotlib import pyplot as plt
from MachineLearning.IO.load_data import LoadData
from MachineLearning.Features.transforms import Transforms
from MachineLearning.Utils.plots import Plots


class Comparison:

    def __init__(self):
        pass

    def compare_filtered_and_unfiltered_eeg(self, result_id: int, channel=1, log_scale=True, y_scale="raw",
                                            same_plot=True):
        """
        Will retrieve the raw and the filtered EEG for given Patient ID and Plot both PSDs to compare filtering.
        :param result_id: Patient ID
        :param channel: Channel of EEG (Options: 1 or 2)
        :param log_scale: Boolean to enable/disable log scaling in plots
        :param y_scale: Value to keep both y_axis in the exact same range.
        Options are 'filtered' for y_axis of filtered EEG plot, 'raw' for y_axis of raw EEG plot and 'min-max' for
        :param same_plot: Boolean to enable/disable plotting both curves into the same plot. Ignores y_scale.
        farthest limits of both EEGs.
        """
        loader = LoadData()
        transforms = Transforms()
        # Load EEGs from same Patient ID to compare
        fs_raw, raw_eegs = loader.return_eeg_tuple(result_id, False)
        fs_filt, filtered_eegs = loader.return_eeg_tuple(result_id, True)

        # Extract EEG of given channel
        raw_eeg = raw_eegs[:, channel - 1]
        filtered_eeg = filtered_eegs[:, channel - 1]

        # Create PSD for both
        raw_freq, raw_power = transforms.return_psd(raw_eeg, fs_raw)
        filt_freq, filt_power = transforms.return_psd(filtered_eeg, fs_filt)

        if same_plot:
            # Create one plot and write second plot into it with different color and label
            fig, ax = Plots.plot_psd(None, raw_freq, raw_power, "raw EEG", log_scale=log_scale)
            # Retrieve y-scale of raw ax scale to use as global ax for both
            if y_scale == "raw":
                raw_ax_y = ax.get_ylim()

            Plots.plot_psd((fig, ax), filt_freq, filt_power, "filtered EEG", "green",
                           f"Power Spectral Density of Patient {result_id}", log_scale)
            ############ Change this later. For now assuming raw y-scale is smaller im scaling it down to it
            if y_scale == "raw":
                ax.set_ylim(raw_ax_y)
        else:
            # Make one Plot with two subplots
            fig, axes = Plots.create_subplot_grid(2)
            raw_ax = axes[0]
            filt_ax = axes[1]
            Plots.plot_psd((fig, raw_ax), raw_freq, raw_power,
                           f"raw_EEG Power Spectral Density of Patient{result_id}", log_scale=log_scale)
            Plots.plot_psd((fig, filt_ax), filt_freq, filt_power,
                           f"filtered_EEG Power Spectral Density of Patient{result_id}", log_scale=log_scale)

            if y_scale == "min-max":
                Plots.align_axis(raw_ax, filt_ax, min_max_scale=True)
            elif y_scale == "raw":
                Plots.align_axis(raw_ax, filt_ax)
            elif y_scale == "filtered":
                Plots.align_axis(filt_ax, raw_ax)
            else:
                raise ValueError(f"Value '{y_scale}' for y_scale doesn't exist. "
                                 f"Valid options are: 'raw', 'filtered' or 'min-max'")

        plt.show()






