from MachineLearning.Models.pca_analyzer import PCAAnalyzer
from MachineLearning.IO.load_data import LoadData
import pandas as pd


def pca_analysis(hyperparameters, class_1, class_0, pca_components=5,
                 outliers: str = None, outlier_run: str = None, model_name: str = None, save_results: bool = True):
    """
    Performs PCA analysis on labeled feature datasets, including handling for outliers.
    Generates visualizations such as 2D and 3D plots of the PCA components, scree plots,
    and retrieves the top feature contributions to the first principal component.

    :param hyperparameters: A dictionary containing hyperparameter settings used to
        load and process data.
    :param class_1: The name or identifier representing the first class of data to be
        analyzed.
    :param class_0: The name or identifier for the second class of data to be analyzed,
        serving as a comparison group.
    :param pca_components: Optional; Number of principal components to compute for the PCA
        analysis. Defaults to 5.
    :param outliers: Optional; String identifier specifying which outlier data should.
        Must be either 'global' or 'local'.
    :param outlier_run: Optional; String identifier for a specific run or instance where the
        outliers were identified. Relevant when outliers are included.
    :param model_name: Optional; String specifying the name of a model, if applicable,
        used in connection with outlier identification or processing.
    :param save_results: Optional; Boolean indicating whether to save the visualizations
        and processed outputs. Defaults to True.
    :return: None
    """
    all_epochs_df, labels = load_labeled_data(hyperparameters, class_1, class_0, outliers, outlier_run, model_name)

    analyzer = PCAAnalyzer(hyperparameters, n_components=pca_components)
    pca_result = analyzer.fit_transform(all_epochs_df)
    print(f"PCA Results:\n {pca_result} \n")

    # 2D-Plot
    analyzer.plot_components_2d(labels=labels, marker_size=5, alpha=0.4, separate_plots=True, save_plot=save_results)
    analyzer.plot_components_2d(labels=labels, marker_size=5, alpha=0.4, separate_plots=False, save_plot=save_results)

    # 3D-Plot
    analyzer.plot_components_3d(labels=labels, marker_size=5, alpha=0.4, separate_plots=True, save_plot=save_results)
    analyzer.plot_components_3d(labels=labels, marker_size=5, alpha=0.4, separate_plots=False, save_plot=save_results)

    # Scree Plot
    analyzer.plot_scree(save_plot=save_results)

    # Feature contributions to PC1
    top_features = analyzer.get_feature_contributions(pc_index=0, top_n=10, save_results=save_results)
    print(f"Top Features PC1:\n {top_features}")
    top_features2 = analyzer.get_feature_contributions(pc_index=1, top_n=10, save_results=save_results)
    print(f"Top Features PC2:\n {top_features2}")

def pca_center_of_cluster_analysis(hyperparameters, class_1, class_0, confidence_intervall, cluster_label,
                                   pca_components=5, outliers: str = None, outlier_run: str = None,
                                   model_name: str = None, save_results: bool = True, dims=2):

    all_epochs_df, labels = load_labeled_data(hyperparameters, class_1, class_0, outliers, outlier_run, model_name)

    analyzer = PCAAnalyzer(hyperparameters, n_components=pca_components)
    pca_result = analyzer.fit_transform(all_epochs_df)
    print(f"PCA Results:\n {pca_result} \n")

    # Region analysis and plot for 2D
    analyzer.get_points_in_region(labels, cluster_label, dims, confidence_intervall, True, save_results)


def return_outliers(outliers: str, loader: LoadData, hyperparameters: dict,
                    outlier_run: str, model_name: str) -> pd.DataFrame:
    """
    This function extracts outlier data based on the specified type (either 'global' or 'local').
    For global outliers, it retrieves data using the provided loader. For local outliers, it identifies
    misclassified epochs from the results and arranges them in a structured DataFrame.

    :param outliers: Specifies the type of outliers to retrieve. Must be either 'global' or 'local'.
    :type outliers: str
    :param loader: Instance of a loader class responsible for loading outlier or result data.
    :type loader: LoadData
    :param hyperparameters: Configuration details required for processing the data.
    :type hyperparameters: dict
    :param outlier_run: Identifier for the specific run from which local outliers are to be fetched.
    :type outlier_run: str
    :param model_name: Name of the model used for generating or predicting data.
    :type model_name: str
    :return: A DataFrame containing the requested outlier data.
    :rtype: pandas.DataFrame
    :raises ValueError: If the 'outliers' parameter is neither 'global' nor 'local'.
    """
    if outliers == "global":
        outliers_df = loader.load_global_outliers(hyperparameters, "epoch")
    elif outliers == "local":
        # Load results for given parameters
        results_df = loader.load_results(hyperparameters, outlier_run, model_name)

        # Get wrongly classified epochs with given label
        misclassified_df = results_df[
            (results_df["label"] == 1) & (results_df["prediction"] != results_df["label"])
            ][["Start", "End", "ResultID"]]
        misclassified_df = pd.DataFrame(misclassified_df)  # Explicit cast, because IDE thinks it's a Series
        outliers_df = misclassified_df.sort_values(by=["ResultID", "Start"]).reset_index(drop=True)
    else:
        raise ValueError("Outliers parameter must be either 'global' or 'local'")

    return outliers_df


def split_by_outliers(df_to_split: pd.DataFrame, outlier_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the given DataFrame into two DataFrames based on whether rows appear in outlier_df.

    :param df_to_split: DataFrame to split with Start, End, ResultID.
    :param outlier_df: DataFrame containing identified outlier epochs with Start, End, ResultID.
    :returns: A tuple of two DataFrames:
              - matched_outlier_df: rows in df_to_split that also appear in outlier_df
              - not_matched_outlier_df: remaining rows in df_to_split
    """
    # Create a unique key for each row to make comparison easier
    df_to_split_keys = df_to_split[['Start', 'End', 'ResultID']].astype(str).agg('_'.join, axis=1)
    outlier_keys = outlier_df[['Start', 'End', 'ResultID']].astype(str).agg('_'.join, axis=1)

    # Mask for match
    is_outlier = df_to_split_keys.isin(set(outlier_keys))

    matched_outlier_df = df_to_split[is_outlier].reset_index(drop=True)
    not_matched_outlier_df = df_to_split[~is_outlier].reset_index(drop=True)

    return matched_outlier_df, not_matched_outlier_df

def load_labeled_data(hyperparameters, class_1, class_0, outliers: str = None, outlier_run: str = None, model_name: str = None):
    """
    Loads labeled data by combining feature data from specified classes and assigning labels.
    Supports normal labeling for provided classes and also handles special labeling for outliers
    if specified. Outliers are assigned a distinct label to differentiate from regular data.

    :param hyperparameters: Parameters used for configuring dataset loading.
    :type hyperparameters: Any
    :param class_1: The identifier for the first class to label in the dataset.
    :type class_1: Any
    :param class_0: The identifier for the second class to label in the dataset.
    :type class_0: Any
    :param outliers: Optional path or identifier to locate outlier data.
        Must be either 'global' or 'local'.
    :type outliers: str, optional
    :param outlier_run: Optional additional specification to process outlier labeling.
    :type outlier_run: str, optional
    :param model_name: Name of the model for which data is being prepared, if applicable.
    :type model_name: str, optional
    :return: A tuple with two elements:
        - The combined labeled dataframe containing all epochs with labels assigned.
        - The corresponding label series if labeling is applied, else None.
    :rtype: tuple[pd.DataFrame, pd.Series or None]
    """
    # Load features and assign labels
    loader = LoadData()
    class_1_df, class_0_df = loader.load_combined_features_df(hyperparameters, class_1, class_0)

    class_0_df["label"] = 0
    # Label outliers if present, else normal labeling of given classes
    if outliers is not None:
        outlier_df = return_outliers(outliers, loader, hyperparameters, outlier_run, model_name)
        class_1_outlier_df, class_1_non_outlier_df = split_by_outliers(class_1_df, outlier_df)
        class_1_non_outlier_df["label"] = 1
        class_1_outlier_df["label"] = 2
        all_epochs_df = pd.concat([class_0_df, class_1_non_outlier_df, class_1_outlier_df], ignore_index=True)

    else:
        class_1_df["label"] = 1
        all_epochs_df = pd.concat([class_0_df, class_1_df], ignore_index=True)

    labels = all_epochs_df["label"] if "label" in all_epochs_df else None

    return all_epochs_df, labels


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

    # pca_center_of_cluster_analysis(hyperparams, class1, class0, 0.95, 1, pca_components=5, outliers="global")
    pca_analysis(hyperparams, class1, class0, outliers="global", outlier_run="norm2_in_place_rm_outlier_6", model_name="svm", save_results=True)
