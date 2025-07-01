"""
This module is to execute all pipeline commands and therefore the main point to run the project.
"""
from MachineLearning.Core.pipeline import Pipeline

INITIAL_DATA_SUBDIR_KEY = "combined_raw_data"
feature_list = ["mean", "variance", "bandpower", "spectral_skewness", "spectral_kurtosis", "shannon_entropy"]
features_to_combine = ["mean", "variance", "bandpower", "spectral_skewness", "spectral_kurtosis", "shannon_entropy"]
test_size = 0.15
random_state = 42


epoch_classes = {0: "faw", 1: "awake"}  # Actual ML Project
# epoch_classes = {0: "normal_an", 1: "awake"}  # Sanity Check


def run():
    """
    Function to execute any code
    """
    # Initialize with key of directory with patient ID subset of interest. Faw and awake flags are default true
    pipeline = Pipeline(INITIAL_DATA_SUBDIR_KEY, epoch_classes)
    # pipeline.raw_eeg_filtering()
    # pipeline.transform_eeg_to_psd()
    # pipeline.feature_extraction(False, feature_list)
    # pipeline.combine_features(False, features_to_combine)
    pipeline.split_classify_evaluate(test_size, random_state, True, C=10, gamma=0.1)


if __name__ == "__main__":
    run()
