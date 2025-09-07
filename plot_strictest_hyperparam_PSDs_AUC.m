%% Clean all and initialize path variables
clear; % Clear workspace
clc;   % Clear command window
aw_filepath = "D:\Daten\Test_and_train\Feature_sets\Awake_20.csv";
faw_filepath = "D:\Daten\Test_and_train\Feature_sets\Feature_sets_70_080_20_5\Summary_Episodes_20_000.csv";

psd_aw_folder = "D:\Daten\Features\PSD\Awake_20";
psd_faw_folder = "D:\Daten\Features\PSD\PSD_70_080_20_5\Summary_Episodes_20_000";

global_outlier_filepath = "D:\Daten\Global_outliers\Global_outliers_70_080_20_5\Summary_Episodes_20_000\global_epoch_outliers.csv";

result_folder_path = "D:\Daten\Further_analysis\PSD\PSD_70_080_20_5\Summary_Episodes_20_000";

%% Plot all combinations of subsets
prepare_and_plot_psd_sets(aw_filepath, faw_filepath, psd_aw_folder, psd_faw_folder, global_outlier_filepath, result_folder_path);
