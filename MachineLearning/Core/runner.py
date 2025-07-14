"""
This module is to execute all pipeline commands and therefore the main point to run the project.
"""
from MachineLearning.Core.pipeline import Pipeline

INITIAL_DATA_SUBDIR_KEY = "combined_raw_data"
feature_list = ["mean", "variance", "bandpower", "spectral_skewness", "spectral_kurtosis", "shannon_entropy"]
features_to_combine = ["mean", "variance", "bandpower", "spectral_skewness", "spectral_kurtosis", "shannon_entropy"]
# feature_list = "all_features"  # If all features should be extracted
test_size = 0.15
random_state = 42

overlaps = [0.0]
min_episode_lengths = [25, 30]


epoch_classes = {0: "faw", 1: "awake"}  # Actual ML Project
# epoch_classes = {0: "normal_an", 1: "awake"}  # Sanity Check


def run(new_params_: dict = None):
    """
    Function to execute any code
    """
    # Initialize with key of directory with patient ID subset of interest. Faw and awake flags are default true
    pipeline = Pipeline(INITIAL_DATA_SUBDIR_KEY, epoch_classes, new_params_, feature_list)
    # pipeline.raw_eeg_filtering()
    pipeline.transform_eeg_to_psd()
    pipeline.feature_extraction()
    pipeline.combine_features(features_to_combine)
    pipeline.split_classify_evaluate(test_size, random_state, True)
    pipeline.analyze_results("svm", ["ResultID"], plots=False)


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
