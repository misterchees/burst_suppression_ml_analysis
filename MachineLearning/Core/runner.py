"""
This module is to execute all pipeline commands and therefore the main point to run the project.
"""
from MachineLearning.Core.pipeline import Pipeline

INITIAL_DATA_SUBDIR_KEY = "combined_raw_data"

# Use None for variable to skip step; Use "all_features" if all features should be used in step
features = ["mean", "variance", "bandpower", "spectral_skewness", "spectral_kurtosis", "shannon_entropy"]
features_to_combine = ["mean", "variance", "bandpower", "spectral_skewness", "spectral_kurtosis", "shannon_entropy"]

# Set dict to None if no extraction AND no combination shall be conduced
features_dict = {
    "features": features,
    "features_to_combine": features_to_combine
}

test_size = 0.15
random_state = 42

overlaps = [0.0]
min_episode_lengths = [10, 15, 20, 25, 30]

epoch_classes = {0: "faw", 1: "awake"}  # Actual ML Project
# epoch_classes = {0: "normal_an", 1: "awake"}  # Sanity Check

model_key = "svm"
filtermethod = "butterworth"


def run(new_params_: dict = None):
    """
    Function to execute any code
    """
    pipeline = Pipeline(
        init_data_key=INITIAL_DATA_SUBDIR_KEY,
        epoch_classes=epoch_classes,
        model_key=model_key,
        filter_method=filtermethod,
        hyperparams=new_params_,
        features_dict=features_dict,
        random_seed=random_state,
        test_size=test_size,
        remove_outliers=True
    )
    # pipeline.raw_eeg_filtering()
    pipeline.transform_eeg_to_psd()
    pipeline.feature_extraction()
    pipeline.combine_features()
    pipeline.split_classify_evaluate()
    pipeline.analyze_results(["ResultID"], plots=False)


if __name__ == "__main__":
    for ep_length in min_episode_lengths:
        for overlap in overlaps:
            param_changes = {
                "current_params": {
                    "fixed_window_size": ep_length,
                    "overlap": overlap
                }
            }
            run(param_changes)
