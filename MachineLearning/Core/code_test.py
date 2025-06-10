from MachineLearning.IO.io_core import IOCore

io_core = IOCore()
feature_names = list(io_core.return_all_feature_keys())
print(feature_names)

psds_folder_name = io_core.return_feature_name("psds")
feature_names.remove(psds_folder_name)
print(feature_names)

# compare plots
# comp = Comparison()
# comp.compare_filtered_and_unfiltered_eeg(10, y_scale="raw")

# pipeline = Pipeline()
# pipeline.feature_extraction(False, "variance", "amplitude", "sample_entropy", "permutation_entropy")

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

# loader = LoadData()
# freq, power = loader.return_raw_eeg_tuple(1)
# filt_freq, filt_power = loader.return_filtered_eeg_tuple(1)

# fig, axes = Plots.create_subplot_grid(2, cols=1, figsize=(10, 6))
# Plots.plot_psd((fig, axes[0]), freq, power)
# Plots.plot_psd((fig, axes[1]), filt_freq, filt_power)

# fig.tight_layout()
# plt.show()
