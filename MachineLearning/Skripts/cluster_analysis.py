from MachineLearning.Models.pca_analyzer import PCAAnalyzer
from MachineLearning.IO.load_data import LoadData
import pandas as pd


def pca_analysis(hyperparameters, class_1, class_0, pca_components=5,
                 outliers: str = None, outlier_run: str = None, model_name: str = None):
    """
    Performs PCA Analysis on labeled Data.
    :param hyperparameters:
    :param class_1:
    :param class_0:
    :param pca_components:
    :param outliers:
    :param outlier_run:
    :param model_name:
    :return:
    """
    # Load features and assign labels
    loader = LoadData()
    class_1_df, class_0_df = loader.load_combined_features_df(hyperparameters, class_1, class_0)

    class_0_df["label"] = 0
    # Label outliers if present, else normal labeling of given classes
    if outliers is not None:
        outlier_df = _return_outliers(outliers, loader, hyperparameters, outlier_run, model_name)
        class_1_outlier_df, class_1_non_outlier_df = _split_by_outliers(class_1_df, outlier_df)
        class_1_non_outlier_df["label"] = 1
        class_1_outlier_df["label"] = 2
        all_epochs_df = pd.concat([class_0_df, class_1_non_outlier_df, class_1_outlier_df], ignore_index=True)

    else:
        class_1_df["label"] = 1
        all_epochs_df = pd.concat([class_0_df, class_1_df], ignore_index=True)

    labels = all_epochs_df["label"] if "label" in all_epochs_df else None

    analyzer = PCAAnalyzer(n_components=pca_components)
    pca_result = analyzer.fit_transform(all_epochs_df)
    print(f"PCA Results:\n {pca_result} \n")

    # 2D-Plot
    analyzer.plot_components_2d(labels=labels)

    # 3D-Plot
    analyzer.plot_components_3d(labels=labels)

    # Scree Plot
    analyzer.plot_scree()

    # Feature contributions to PC1
    top_features = analyzer.get_feature_contributions(pc_index=0, top_n=10)
    print(f"Top Features:\n {top_features}")


def _return_outliers(outliers, loader, hyperparameters, outlier_run, model_name) -> pd.DataFrame:
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


def _split_by_outliers(awake_df: pd.DataFrame, outlier_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the awake_df into two DataFrames based on whether rows appear in outlier_df.

    :param awake_df: DataFrame containing all awake epochs with Start, End, ResultID.
    :param outlier_df: DataFrame containing identified outlier epochs with Start, End, ResultID.
    :returns: A tuple of two DataFrames:
              - matched_outlier_df: rows in awake_df that also appear in outlier_df
              - not_matched_outlier_df: remaining rows in awake_df
    """
    # Create a unique key for each row to make comparison easier
    awake_keys = awake_df[['Start', 'End', 'ResultID']].astype(str).agg('_'.join, axis=1)
    outlier_keys = outlier_df[['Start', 'End', 'ResultID']].astype(str).agg('_'.join, axis=1)

    # Mask for match
    is_outlier = awake_keys.isin(set(outlier_keys))

    matched_outlier_df = awake_df[is_outlier].reset_index(drop=True)
    not_matched_outlier_df = awake_df[~is_outlier].reset_index(drop=True)

    return matched_outlier_df, not_matched_outlier_df


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

    pca_analysis(hyperparams, class1, class0, outliers="global")