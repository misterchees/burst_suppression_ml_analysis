from MachineLearning.IO.load_data import LoadData, load_psd_with_start_end_resultid
from MachineLearning.IO.save_result import SaveResult
from MachineLearning.Skripts.cluster_analysis import return_outliers, split_by_outliers
from MachineLearning.Utils.plots import Plots
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def calculate_mean_psds(hyperparameters: dict, class_1: str, class_0: str, plot: bool = True, save_results: bool = True):
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

    spread_metric = "std"
    av_class_0_df = average_psd_from_epochs(psd_faw_folderpath, class_0_df, spread_metric=spread_metric)
    av_class_1_outlier_df = average_psd_from_epochs(psd_awake_folderpath, class_1_outlier_df, spread_metric=spread_metric)
    av_class_1_non_outlier_df = average_psd_from_epochs(psd_awake_folderpath, class_1_non_outlier_df, spread_metric=spread_metric)

    metric_name = "standarderror" if spread_metric == "sem" else "standarddeviation"
    if save_results:
        saver = SaveResult()
        saver.save_further_analysis(hyperparameters, av_class_0_df, "dataframe", "psd",
                                    "PSDs_faw_average", f"uncertainty_metric_{metric_name}")
        saver.save_further_analysis(hyperparameters, av_class_1_outlier_df, "dataframe", "psd",
                                    "PSDs_wrong_awake_average", f"uncertainty_metric_{metric_name}")
        saver.save_further_analysis(hyperparameters, av_class_1_non_outlier_df, "dataframe", "psd",
                                    "PSDs_correct_awake_average", f"uncertainty_metric_{metric_name}")

    if plot:
        _plot_3_psds(av_class_0_df, av_class_1_outlier_df, av_class_1_non_outlier_df, hyperparameters, save=False,
                     title_suffix=f"uncertainty_metric_{metric_name}")
        if save_results:
            _plot_3_psds(av_class_0_df, av_class_1_outlier_df, av_class_1_non_outlier_df, hyperparameters, save=True,
                         title_suffix=f"uncertainty_metric_{metric_name}")

def calculate_center_of_mass_psds(hyperparameters: dict, confidence: float,
                                  plot: bool = True, save_results: bool = True):
    loader = LoadData()
    confidence_str = str(confidence).replace(".", "")
    faw_file_name = f"PCA_clusterlabel_0_region_with_confidence_{confidence_str}_dims_2.csv"
    wrong_awake_file_name = f"PCA_clusterlabel_2_region_with_confidence_{confidence_str}_dims_2.csv"
    faw_df = loader.load_further_results(hyperparameters, "pca", "dataframe",faw_file_name)
    wrong_awake_df = loader.load_further_results(hyperparameters, "pca", "dataframe",wrong_awake_file_name)

    psd_faw_folderpath = loader.return_file_fullpath(
        hyperparameters, False, False, "faw", ["features", "psds"])

    psd_awake_folderpath = loader.return_file_fullpath(
        hyperparameters, False, False, "awake", ["features", "psds"])

    spread_metric = "std"
    av_faw_df = average_psd_from_epochs(psd_faw_folderpath, faw_df, spread_metric=spread_metric)
    av_wrong_awake_df = average_psd_from_epochs(psd_awake_folderpath, wrong_awake_df, spread_metric=spread_metric)

    metric_name = "standarderror" if spread_metric == "sem" else "standarddeviation"
    if save_results:
        saver = SaveResult()
        saver.save_further_analysis(hyperparameters, av_faw_df, "dataframe", "psd",
                                    "PSDs_faw_com_average", f"uncertainty_metric_{metric_name}")
        saver.save_further_analysis(hyperparameters, av_wrong_awake_df, "dataframe", "psd",
                                    "PSDs_wrong_awake_com_average", f"uncertainty_metric_{metric_name}")

    if plot:
        _plot_2_center_of_mass_psds(av_faw_df, av_wrong_awake_df, hyperparameters, save=False,
                     title_suffix=f"uncertainty_metric_{metric_name}")
        if save_results:
            _plot_2_center_of_mass_psds(av_faw_df, av_wrong_awake_df, hyperparameters, save=True,
                                        title_suffix=f"uncertainty_metric_{metric_name}")



def average_psd_from_epochs(psd_folderpath: str, epochs_df: pd.DataFrame, spread_metric="std") -> pd.DataFrame:
    """
    Calculates the average PSD and an uncertainty metric from multiple PSD epochs.

    :param psd_folderpath: Path to the PSD CSV files.
    :param epochs_df: DataFrame containing Start, End, and ResultID columns.
    :param spread_metric: "std" for standard deviation, "sem" for standard error of the mean.
    :return: DataFrame with Frequency_Hz, PSD_mean, and PSD_spread columns.
    """
    psd_values = []
    freq_axis = None

    for _, row in epochs_df.iterrows():
        psd_filename = f"PSD_{int(row['Start'])}_{int(row['End'])}_{int(row['ResultID'])}.csv"
        psd_df, *_ = load_psd_with_start_end_resultid(psd_folderpath, psd_filename)

        if psd_df is not None and not psd_df.empty:
            if freq_axis is None:
                freq_axis = psd_df["Frequency_Hz"].values
            else:
                if not np.all(psd_df["Frequency_Hz"].values == freq_axis):
                    raise ValueError(f"Frequency axis differs in file: {psd_filename}")
            psd_values.append(psd_df["PSD_V2_per_Hz"].values)

    if not psd_values:
        raise ValueError("No matching PSD data found.")

    psd_array = np.array(psd_values)  # shape: (n_epochs, n_freqs)
    avg_psd_values = np.mean(psd_array, axis=0)

    if spread_metric == "std":
        spread_values = np.std(psd_array, axis=0)
    elif spread_metric == "sem":
        spread_values = np.std(psd_array, axis=0) / np.sqrt(psd_array.shape[0])
    else:
        raise ValueError("spread_metric must be 'std' or 'sem'.")

    avg_psd_df = pd.DataFrame({
        "Frequency_Hz": freq_axis,
        "PSD_V2_per_Hz_mean": avg_psd_values,
        "PSD_V2_per_Hz_spread": spread_values
    })

    return avg_psd_df


def _plot_3_psds(av_class_0_df: pd.DataFrame, av_class_1_outlier_df: pd.DataFrame, av_class_1_non_outlier_df: pd.DataFrame,
                 hyperparameters: dict, save: bool, title_suffix: str = ""):
    fig, ax = Plots.plot_psd(None, av_class_0_df["Frequency_Hz"], av_class_0_df["PSD_V2_per_Hz_mean"],
                             "faw_average", log_scale=True, spread=av_class_0_df["PSD_V2_per_Hz_spread"])
    Plots.plot_psd((fig, ax), av_class_1_outlier_df["Frequency_Hz"], av_class_1_outlier_df["PSD_V2_per_Hz_mean"],
                   "wrong_awake_average", log_scale=True, color="red", spread=av_class_1_outlier_df["PSD_V2_per_Hz_spread"])
    fig, ax = Plots.plot_psd((fig, ax), av_class_1_non_outlier_df["Frequency_Hz"], av_class_1_non_outlier_df["PSD_V2_per_Hz_mean"],
                    "correct_awake_average", log_scale=True, color="green", spread=av_class_1_non_outlier_df["PSD_V2_per_Hz_spread"],
                             title=f"PSD_averages_comparison_{title_suffix}")

    if save:
        saver = SaveResult()
        saver.save_further_analysis(hyperparameters, fig, "plot", "psd", "PSDs",f"averages_comparison_{title_suffix}")
    else:
        plt.show()

def _plot_2_center_of_mass_psds(av_class_0_df: pd.DataFrame, av_class_2_df: pd.DataFrame, hyperparameters: dict,
                                save: bool, title_suffix: str = ""):
    fig, ax = Plots.plot_psd(None, av_class_0_df["Frequency_Hz"], av_class_0_df["PSD_V2_per_Hz_mean"],
                             "faw_com_average", log_scale=True, spread=av_class_0_df["PSD_V2_per_Hz_spread"])
    fig, ax = Plots.plot_psd((fig, ax), av_class_2_df["Frequency_Hz"], av_class_2_df["PSD_V2_per_Hz_mean"],
                   "wrong_awake_com_average", log_scale=True, color="red", spread=av_class_2_df["PSD_V2_per_Hz_spread"],
                             title=f"PSD_center_of_mass_averages_comparison_{title_suffix}")

    if save:
        saver = SaveResult()
        saver.save_further_analysis(hyperparameters, fig, "plot", "psd", "PSDs",f"com_averages_comparison_{title_suffix}")
    else:
        plt.show()


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

    # calculate_mean_psds(hyperparams, class1, class0, plot=True, save_results=True)
    calculate_center_of_mass_psds(hyperparams, 0.25, plot=True, save_results=True)