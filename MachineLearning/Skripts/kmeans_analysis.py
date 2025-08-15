from MachineLearning.Models.k_means import KMeans
from MachineLearning.IO.load_data import LoadData
from MachineLearning.Skripts.cluster_analysis import return_outliers, split_by_outliers
import pandas as pd



def analyze_data(hyperparameters: dict, n_cluster: int, _random_state: int = 42, class_1: str = "awake",
                 class_0: str = "faw", plot: str = "2D", _data_subset: str = "all"):
    # Load combined feature dfs to get all epochs from this set
    loader = LoadData()
    class_1_df, class_0_df = loader.load_combined_features_df(hyperparameters, class_1, class_0)

    if _data_subset == "all":
        df_to_analyze = pd.concat([class_0_df, class_1_df], ignore_index=True)
    elif _data_subset == "faw":
        df_to_analyze = class_0_df
    elif _data_subset == "awake":
        df_to_analyze = class_1_df
    elif _data_subset == "correct_awake" or _data_subset == "wrong_awake":
        # Load global outlier epochs and split awake data accordingly
        outlier_df = return_outliers("global", loader, hyperparameters, outlier_run="", model_name="")
        class_1_outlier_df, class_1_non_outlier_df = split_by_outliers(class_1_df, outlier_df)
        df_to_analyze = class_1_outlier_df if _data_subset == "wrong_awake" else class_1_non_outlier_df
    else:
        raise ValueError(f"Invalid data subset specified: {_data_subset}.")

    _run_analysis(df_to_analyze.drop(columns=["Start", "End", "ResultID"]).values, n_cluster, _random_state, plot, hyperparameters, _data_subset)





def _run_analysis(data, n_cluster: int, _random_state: int, plot: str, hyperparameters:dict, _data_subset:str):
    data = pd.DataFrame(data)

    model = KMeans(hyperparameters, n_clusters=n_cluster, random_state=_random_state)
    labels = model.fit_predict(data)

    if plot == "2D" or plot == "3D":
        if plot == "2D":
            model.plot_components_2d(data,labels=labels, marker_size=5, alpha=0.4, separate_plots=False, save_plot=True,
                           title=f"Kmeans_2D_clusters_{n_cluster}_random_state_{_random_state}_for_{_data_subset}")
        else:
            model.plot_components_3d(data,labels=labels, marker_size=5, alpha=0.4, separate_plots=False, save_plot=True,
                           title=f"Kmeans_3D_clusters_{n_cluster}_random_state_{_random_state}_for_{_data_subset}")


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
    random_states = [42, 89, 111, 4]
    n_cluster_array = [2, 3, 4, 5]
    data_subset_array = ["correct_awake", "all"]
    for data_subset in data_subset_array:
        for n_clusters in n_cluster_array:
            for random_state in random_states:
                print(f"Running analysis for {n_clusters} clusters for data subset {data_subset}.")
                analyze_data(hyperparams, n_clusters, class_1="awake", class_0="faw", plot="2D", _data_subset=data_subset, _random_state=random_state)
