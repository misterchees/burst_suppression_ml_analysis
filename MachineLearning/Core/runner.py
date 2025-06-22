"""
This module is to execute all pipeline commands and therefore the main point to run the project.
"""
from MachineLearning.Core.pipeline import Pipeline

INITIAL_DATA_SUBDIR_KEY = "combined_raw_data"

epoch_classes = {0: "faw", 1: "awake"}  # Actual ML Project
# epoch_classes = {0: "faw", 1: "normal_an"}  # Sanity Check


def run():
    """
    Function to execute any code
    """
    # Initialize with key of directory with patient ID subset of interest. Faw and awake flags are default true
    pipeline = Pipeline(INITIAL_DATA_SUBDIR_KEY, *epoch_classes.values())
    # pipeline.raw_eeg_filtering()
    # pipeline.transform_eeg_to_psd()
    # pipeline.feature_extraction(False, "mean", "variance", "bandpower", "spectral_skewness",
    #                            "spectral_kurtosis", "shannon_entropy")
    # pipeline.combine_all_features()
    # pipeline.combine_features("mean", "variance", "bandpower", "spectral_skewness",
    #                          "spectral_kurtosis", "shannon_entropy")
    train_path, test_path = pipeline.create_splits(epoch_classes[0], epoch_classes[1], .15, 42)
    y_pred, y_test, y_proba = pipeline.run_svm_classifier(train_path, test_path)
    pipeline.evaluate_metrics(y_pred, y_test, y_proba)


if __name__ == "__main__":
    run()
