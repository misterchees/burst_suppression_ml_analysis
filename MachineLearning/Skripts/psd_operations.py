from MachineLearning.IO.load_data import LoadData, load_psd_with_start_end_resultid
from MachineLearning.IO.save_result import SaveResult
from MachineLearning.Skripts.cluster_analysis import return_outliers, split_by_outliers
from MachineLearning.Utils.plots import Plots
import pandas as pd
from matplotlib import pyplot as plt


def calculate_mean_psds(hyperparameters: dict, class_1: str, class_0: str, plot: bool = True):
    # Load combined feature dfs to get all epochs from this set
    loader = LoadData()
    class_1_df, class_0_df = loader.load_combined_features_df(hyperparameters, class_1, class_0)

    # Load global outlier epochs and split awake data accordingly
    outlier_df = return_outliers("global", loader, hyperparameters, outlier_run="", model_name="")
    class_1_outlier_df, class_1_non_outlier_df = split_by_outliers(class_1_df, outlier_df)

    psd_faw_folderpath = loader.return_file_fullpath(
        hyperparameters, False, False, "faw",["features", "psds"])

    psd_awake_folderpath = loader.return_file_fullpath(
        hyperparameters, False, False, "awake",["features", "psds"])

    av_class_0_df = average_psd_from_epochs(psd_faw_folderpath, class_0_df)
    av_class_1_outlier_df = average_psd_from_epochs(psd_awake_folderpath, class_1_outlier_df)
    av_class_1_non_outlier_df = average_psd_from_epochs(psd_awake_folderpath, class_1_non_outlier_df)

    if plot:
        fig, ax = Plots.plot_psd(None, av_class_0_df["Frequency_Hz"], av_class_0_df["PSD_V2_per_Hz"],
                                 "faw_average", log_scale=True)
        Plots.plot_psd((fig, ax), av_class_1_outlier_df["Frequency_Hz"], av_class_1_outlier_df["PSD_V2_per_Hz"],
                       "wrong_awake_average", log_scale=True, color="red")
        Plots.plot_psd((fig, ax), av_class_1_non_outlier_df["Frequency_Hz"], av_class_1_non_outlier_df["PSD_V2_per_Hz"],
                       "correct_awake_average", log_scale=True, color="green")
        plt.show()



def average_psd_from_epochs(psd_folderpath: str, epochs_df: pd.DataFrame) -> pd.DataFrame:
    psd_values = []
    freq_axis = None

    for _, row in epochs_df.iterrows():
        psd_filename = f"PSD_{int(row['Start'])}_{int(row['End'])}_{int(row['ResultID'])}.csv"
        psd_df, *_ = load_psd_with_start_end_resultid(psd_folderpath, psd_filename)

        if psd_df is not None and not psd_df.empty:
            if freq_axis is None:
                freq_axis = psd_df["Frequency_Hz"].values
            else:
                if not (psd_df["Frequency_Hz"].values == freq_axis).all():
                    raise ValueError(f"Frequency axis has different values then all others: {psd_filename}")
            psd_values.append(psd_df["PSD_V2_per_Hz"].values)

    if not psd_values:
        raise ValueError("No matching PSD data found in given directory for given epochs.")

    # Average of all power values
    avg_psd_values = sum(psd_values) / len(psd_values)

    # Create average dataframe
    avg_psd_df = pd.DataFrame({
        "Frequency_Hz": freq_axis,
        "PSD_V2_per_Hz": avg_psd_values
    })

    return avg_psd_df

if __name__ == "__main__":
    hyperparams = {
        "merged_episodes": False,
        "bis_threshold": 70,
        "mac_threshold": 0.8,
        "min_episode_length": 20,
        "refractory_time": 5,
        "fixed_window_size": 20,
        "overlap": 0.0
    }

    class1 = "awake"
    class0 = "faw"

    calculate_mean_psds(hyperparams, class1, class0)