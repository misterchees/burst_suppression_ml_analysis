"""
This module is to execute all pipeline commands and therefore the main point to run the project.
"""
from MachineLearning.Core.pipeline import Pipeline

INITIAL_DATA_SUBDIR_KEY = "combined_raw_data"
channel = 1
model_key = "svm"
model_params = {"C": 1, "kernel": "rbf"}
filter_method = "butterworth"
transform_method = "welch"
run_name = "baseline_run_rm_ids_5"

all_run_params_dict = {
    "current_params": {
        "merged_episodes": False,
        "bis_threshold": 70,
        "mac_threshold": 0.8,
        "min_episode_length": 20,
        "refractory_time": 5,
        "fixed_window_size": 20,
        "overlap": 0.0
    },
    "filtering_params": {
        filter_method: {"lowcut": 0.5, "highcut": 30.0, "order": 4}
    },
    "transform_params": {
        transform_method: {"channel": channel, "nperseg_seconds": 2, "fs": 128}
    },
    "feature_params": {
        "relative_bandpower": {"normalize_to": "bands"},
        "shannon_entropy": {"normalize": True},
        "spectral_skewness": {"normalize": True, "n_method": "clip", "lower_bound": 0, "upper_bound": 1},
        "spectral_kurtosis": {"normalize": True, "n_method": "clip", "lower_bound": 0, "upper_bound": 1},
        "mean": {"channel": channel},
        "variance": {"channel": channel},
        "amplitude": {"channel": channel},
        "sample_entropy": {"channel": channel, "emb_dim": 2, "tolerance": 0.2},
        "permutation_entropy": {"channel": channel, "order": 3, "delay": 1, "normalize": True},
        "fuzzy_entropy": {"channel": channel, "m": 2, "r": 0.2, "n": 2}
    },
    "classification_params": {
        "test_size": 0.15,
        "random_seed": 42,
        "remove_outliers": True,
        "remove_outlier_epochs": False,
        "outlier_run_name": "baseline_run_rm_ids_4",
        model_key: model_params
    }
}

# Use None for variable to skip step; Use "all_features" if all features should be used in step
features = ["mean", "variance", "bandpower", "spectral_skewness", "spectral_kurtosis", "shannon_entropy", "permutation_entropy"]
features_to_combine = ["mean", "variance", "bandpower", "spectral_skewness", "spectral_kurtosis", "shannon_entropy", "permutation_entropy"]

# Set dict to None if no extraction AND no combination shall be conduced
features_dict = {
    "features": features,
    "features_to_combine": features_to_combine
}

# Metadata to analyze for errors
metadata_to_analyze = ["ResultID"]

epoch_classes = {0: "faw", 1: "awake"}  # Actual ML Project
# epoch_classes = {0: "normal_an", 1: "awake"}  # Sanity Check

steps_of_workflow = ["transform", "extract", "combine", "classify", "analyze"]


def run():
    """
    Function to execute any code
    """
    pipeline = Pipeline(
        init_data_key=INITIAL_DATA_SUBDIR_KEY,
        epoch_classes=epoch_classes,
        update_dict=all_run_params_dict,
        filter_method=filter_method,
        model_key=model_key,
        transform_method=transform_method,
        features_dict=features_dict,
        metadata_to_analyze=metadata_to_analyze,
        run_name=run_name,
        force_overwrite=True,
        global_outliers=True
    )

    pipeline.complete_run(steps_of_workflow)


if __name__ == "__main__":
    run()
