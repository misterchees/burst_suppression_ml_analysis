from matplotlib import pyplot as plt
from MachineLearning.Utils.plots import Plots
from MachineLearning.Preprocessing.filtering import Filtering
from MachineLearning.Utils.config_loader import load_config
from MachineLearning.IO.load_data import LoadData

# orders = [2, 4, 6, 8, 10]
# fig, axes = Plots.create_subplot_grid(len(orders), cols=1, figsize=(10, 3 * len(orders)))
# for idx, order in enumerate(orders):
#     cur_fig_and_ax = (fig, axes[idx])
#     Plots.plot_butterworth_filtering(fig_and_ax=cur_fig_and_ax, order=orders[idx])
#
# fig.tight_layout()
# plt.show()

# filtering = Filtering()
# filtering.butterworth(1)

loader = LoadData()
freq, power = loader.return_raw_eeg_tuple(1)
filt_freq, filt_power = loader.return_filtered_eeg_tuple(1)

# fig, axes = Plots.create_subplot_grid(2, cols=1, figsize=(10, 6))
# Plots.plot_psd((fig, axes[0]), freq, power)
# Plots.plot_psd((fig, axes[1]), filt_freq, filt_power)

# fig.tight_layout()
# plt.show()
