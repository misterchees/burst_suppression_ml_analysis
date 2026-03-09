from pathlib import Path

from MachineLearning.IO.load_data import LoadData, load_psd_with_start_end_resultid
from MachineLearning.IO.save_result import SaveResult
from MachineLearning.Scripts.ResultAnalyses.cluster_analysis import return_outliers, split_by_outliers
from MachineLearning.Utils.path_manager import PathManager
from MachineLearning.Utils.plots import Plots
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

pm = PathManager()
loader = LoadData(pm)
saver = SaveResult(pm)

def calculate_mean_psds(hyperparameters: dict, class_1: str, class_0: str, plot: bool = True, save_results: bool = True, outliers: bool = False):
    # Load combined feature dfs to get all epochs from this set

    class_1_df, class_0_df = loader.load_combined_features_df(hyperparameters, class_1, class_0)

    if outliers:
        # Load global outlier epochs and split awake data accordingly
        outlier_df = return_outliers("global", hyperparameters, outlier_run="", model_name="")
        class_1_outlier_df, class_1_non_outlier_df = split_by_outliers(class_1_df, outlier_df)

    psd_faw_folderpath = pm.resolve_episode_path(
        hyperparameters, "faw",["features", "psds"], False, False
    )

    psd_awake_folderpath = pm.resolve_episode_path(
        hyperparameters, "awake",["features", "psds"], False, False
    )

    spread_metric = "sem"
    av_class_0_df = average_psd_from_epochs(psd_faw_folderpath, class_0_df, spread_metric=spread_metric)
    if outliers:
        av_class_1_outlier_df = average_psd_from_epochs(psd_awake_folderpath, class_1_outlier_df, spread_metric=spread_metric)
        av_class_1_non_outlier_df = average_psd_from_epochs(psd_awake_folderpath, class_1_non_outlier_df, spread_metric=spread_metric)
    else:
        av_class_1_df = average_psd_from_epochs(psd_awake_folderpath, class_1_df, spread_metric=spread_metric)

    metric_name = "standarderror" if spread_metric == "sem" else "standarddeviation"
    if save_results:
        folder_path = pm.get_complex_ml_path(hyperparameters, ["further_analysis", "psd"], False, True)
        # Save class 0 df
        file_suffix = f"uncertainty_metric_{metric_name}"
        saver.save_file("dataframe", folder_path, "PSDs_faw_average", file_suffix, av_class_0_df)
        if outliers:
            # Save class 1 outlier df
            saver.save_file(
                "dataframe", folder_path, "PSDs_wrong_awake_average", file_suffix, av_class_1_outlier_df
            )
            # Save class 1 non outlier df
            saver.save_file(
                "dataframe", folder_path, "PSDs_correct_awake_average", file_suffix, av_class_1_non_outlier_df
            )
        else:
            # Save class 1 df
            saver.save_file(
                "dataframe", folder_path, "PSDs_awake_average", file_suffix, av_class_1_df
            )

    if plot:
        if outliers:
            _plot_3_psds(av_class_0_df, av_class_1_outlier_df, av_class_1_non_outlier_df, hyperparameters, save=False)
            if save_results:
                _plot_3_psds(av_class_0_df, av_class_1_outlier_df, av_class_1_non_outlier_df, hyperparameters, save=True)
        else:
            _plot_2_psds(av_class_0_df, av_class_1_df, hyperparameters, save=False)
            if save_results:
                _plot_2_psds(av_class_0_df, av_class_1_df, hyperparameters, save=True)

def calculate_center_of_mass_psds(hyperparameters: dict, confidence: float, class_a: int, class_b: int,
                                  plot: bool = True, save_results: bool = True, spread_metric="sem"):
    """class_a should always be faw, class_b can be wrong_awake or correct_awake, confidence should be between 0 and 1."""

    av_faw_df, av_awake_df = load_pca_cluster_center_results(hyperparameters, confidence, class_a, class_b, spread_metric=spread_metric)

    metric_name = "standarderror" if spread_metric == "sem" else "standarddeviation"
    if save_results:
        folder_path = pm.get_complex_ml_path(hyperparameters, ["further_analysis", "psd"], False, True)
        file_suffix = f"uncertainty_metric_{metric_name}"
        saver.save_file("dataframe", folder_path, "PSDs_faw_com_average", file_suffix, av_faw_df)


        pref_str = "correct" if class_b == 1 else "wrong"
        saver.save_file("dataframe", folder_path, f"PSDs_{pref_str}_awake_com_average", file_suffix, av_awake_df)

    if plot:
        _plot_2_center_of_mass_psds(av_faw_df, av_awake_df, hyperparameters, save=False,
                     title_suffix=f"uncertainty_metric_{metric_name}", class_a=class_a, class_b=class_b)
        if save_results:
            _plot_2_center_of_mass_psds(av_faw_df, av_awake_df, hyperparameters, save=True,
                                        title_suffix=f"uncertainty_metric_{metric_name}", class_a=class_a, class_b=class_b)

def load_pca_cluster_center_results(hyperparameters: dict, confidence: float, class_a: int, class_b: int, spread_metric="std"):
    confidence_str = str(confidence).replace(".", "")
    a_file_name = f"PCA_clusterlabel_{class_a}_region_with_confidence_{confidence_str}_dims_2.csv"
    b_file_name = f"PCA_clusterlabel_{class_b}_region_with_confidence_{confidence_str}_dims_2.csv"
    a_df = loader.load_further_results(hyperparameters, "pca", a_file_name)
    b_df = loader.load_further_results(hyperparameters, "pca", b_file_name)

    psd_faw_folderpath = pm.resolve_episode_path(
        hyperparameters, "faw", ["features", "psds"], False, False
    )

    psd_awake_folderpath = pm.resolve_episode_path(
        hyperparameters, "awake", ["features", "psds"], False, False
    )


    if class_a == 0:
        av_faw_df = average_psd_from_epochs(psd_faw_folderpath, a_df, spread_metric=spread_metric)
    else:
        av_faw_df = average_psd_from_epochs(psd_awake_folderpath, a_df, spread_metric=spread_metric)
    if class_b == 0:
        av_awake_df = average_psd_from_epochs(psd_faw_folderpath, b_df, spread_metric=spread_metric)
    else:
        av_awake_df = average_psd_from_epochs(psd_awake_folderpath, b_df, spread_metric=spread_metric)
    return av_faw_df, av_awake_df


def average_psd_from_epochs(psd_folderpath: Path, epochs_df: pd.DataFrame, spread_metric="std") -> pd.DataFrame:
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

def plot_single_psd(patient_id: int, filtered: bool):

    # Import and initialize classes
    from MachineLearning.Features.transforms import Transforms
    psd_transforms = Transforms(pm, ("faw","awake"), "welch", None)
    plotter = Plots()

    # Get EEG -> calculate PSD from channel 1 -> plot PSD
    fs, patient_eeg = loader.load_eeg_data(patient_id, filtered)
    frequency, power = psd_transforms.calculate_psd(patient_eeg[:, 0])
    plotter.plot_psd(None, frequency, power, None, log_scale=True, max_freq=30, min_power=0.000000001 )
    plt.show()

def _plot_3_psds(av_class_0_df: pd.DataFrame, av_class_1_outlier_df: pd.DataFrame, av_class_1_non_outlier_df: pd.DataFrame,
                 hyperparameters: dict, save: bool, title_suffix: str = "", max_freq: float = 30, min_power: float = 0.0001,):

    fig, ax = Plots.plot_psd(None, av_class_0_df["Frequency_Hz"], av_class_0_df["PSD_V2_per_Hz_mean"],
                             "FAW", log_scale=True, spread=av_class_0_df["PSD_V2_per_Hz_spread"] ,
                             max_freq=max_freq, min_power=min_power)
    Plots.plot_psd((fig, ax), av_class_1_outlier_df["Frequency_Hz"], av_class_1_outlier_df["PSD_V2_per_Hz_mean"],
                   "MAW", log_scale=True, color="red", spread=av_class_1_outlier_df["PSD_V2_per_Hz_spread"] ,
                   max_freq=max_freq, min_power=min_power)
    fig, ax = Plots.plot_psd((fig, ax), av_class_1_non_outlier_df["Frequency_Hz"], av_class_1_non_outlier_df["PSD_V2_per_Hz_mean"],
                    "CAW", log_scale=True, color="green", spread=av_class_1_non_outlier_df["PSD_V2_per_Hz_spread"],
                             title=f"Mean PSD Comparison (epoch-wise Normalization) {title_suffix}" , max_freq=max_freq, min_power=min_power)

    if save:
        folder_path = pm.get_complex_ml_path(hyperparameters, ["further_analysis", "psd"], False, True)
        file_suffix = f"averages_comparison_{title_suffix}"
        saver.save_file("plot", folder_path, "PSDs", file_suffix, fig)
    else:
        plt.show()

def _plot_2_psds(av_class_0_df: pd.DataFrame, av_class_1_df: pd.DataFrame,
                 hyperparameters: dict, save: bool, title_suffix: str = "", max_freq: float = 30, min_power: float = 0.0001):

    fig, ax = Plots.plot_psd(None, av_class_0_df["Frequency_Hz"], av_class_0_df["PSD_V2_per_Hz_mean"],
                             "FAW", log_scale=True, spread=av_class_0_df["PSD_V2_per_Hz_spread"],
                             max_freq=max_freq, min_power=min_power)
    fig, ax = Plots.plot_psd((fig, ax), av_class_1_df["Frequency_Hz"], av_class_1_df["PSD_V2_per_Hz_mean"],
                    "AW", log_scale=True, color="green", spread=av_class_1_df["PSD_V2_per_Hz_spread"],
                             title=f"Mean PSD Comparison {title_suffix}", max_freq=max_freq, min_power=min_power)

    if save:
        folder_path = pm.get_complex_ml_path(hyperparameters, ["further_analysis", "psd"], False, True)
        file_suffix = f"averages_comparison_{title_suffix}"
        saver.save_file("plot", folder_path, "PSDs", file_suffix, fig)
    else:
        plt.show()


def _plot_2_center_of_mass_psds(av_class_a_df: pd.DataFrame, av_class_b_df: pd.DataFrame, hyperparameters: dict,
                                save: bool, class_a: int, class_b: int, title_suffix: str = "", max_freq: float = 30,
                                min_power: float = 0.0001):
    class_dict = {0:"faw", 1:"correct_awake", 2:"wrong_awake"}
    a_name = class_dict[class_a]
    b_name = class_dict[class_b]
    fig, ax = Plots.plot_psd(None, av_class_a_df["Frequency_Hz"], av_class_a_df["PSD_V2_per_Hz_mean"],
                             f"{a_name}_com_average", log_scale=True, spread=av_class_a_df["PSD_V2_per_Hz_spread"],
                             max_freq=max_freq, min_power=min_power)
    fig, ax = Plots.plot_psd((fig, ax), av_class_b_df["Frequency_Hz"], av_class_b_df["PSD_V2_per_Hz_mean"],
                   f"{b_name}_com_average", log_scale=True, color="red", spread=av_class_b_df["PSD_V2_per_Hz_spread"],
                             title=f"PSD_center_of_mass_averages_comparison_{title_suffix}",
                             max_freq=max_freq, min_power=min_power)

    if save:
        folder_path = pm.get_complex_ml_path(hyperparameters, ["further_analysis", "psd"], False, True)
        file_suffix = f"com_{a_name}_vs_{b_name}_averages_comparison_{title_suffix}"
        saver.save_file("plot", folder_path, "PSDs", file_suffix, fig)
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

    # calculate_mean_psds(hyperparams, class1, class0, plot=True, save_results=True, outliers=False)
    # calculate_center_of_mass_psds(hyperparams, 0.25,1,2, plot=True, save_results=True, spread_metric="sem")
    plot_single_psd(1127, True)