from MachineLearning.IO.load_data import LoadData
from MachineLearning.Utils.plots import Plots
loader = LoadData()
plotter = Plots()


_patient_id = 6
_filtered = False
_channel = 1
_both_channels = True


def plot_EEG(patient_id: int, filtered: bool, channel: int, both_channels: bool = False):
    fs, all_channel_eeg = loader.return_eeg_tuple(patient_id, filtered)
    if both_channels:
        plotter.plot_two_channel_eeg(all_channel_eeg, fs, filtered)
    else:
        one_channel_eeg = all_channel_eeg[:, channel - 1]
        plotter.plot_eeg_signal(one_channel_eeg, fs, filtered)


if __name__ == "__main__":
    plot_EEG(_patient_id, _filtered, _channel, _both_channels)