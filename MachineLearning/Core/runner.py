"""
This module is to execute all pipeline commands and therefore the main point to run the project.
"""
from MachineLearning.Core.pipeline import Pipeline

INITIAL_DATA_SUBDIR_KEY = "combined_raw_data"

epoch_classes = {0: "faw", 1: "awake"}  # Actual ML Project
# epoch_classes = {0: "normal_an", 1: "awake"}  # Sanity Check
class_0 = epoch_classes[0]
class_1 = epoch_classes[1]


def run():
    """
    Function to execute any code
    """
    # Initialize with key of directory with patient ID subset of interest. Faw and awake flags are default true
    pipeline = Pipeline(INITIAL_DATA_SUBDIR_KEY, class_0, class_1)
    # pipeline.raw_eeg_filtering()
    # pipeline.transform_eeg_to_psd()
    # pipeline.feature_extraction(False, "mean", "variance", "bandpower", "spectral_skewness",
    #                            "spectral_kurtosis", "shannon_entropy")
    # pipeline.combine_all_features()
    # pipeline.combine_features(False, "mean", "variance", "bandpower", "spectral_skewness",
    #                          "spectral_kurtosis", "shannon_entropy")
    train_path, test_path = pipeline.create_splits(class_0, class_1, .15, 42)
    y_pred, y_test, y_proba = pipeline.run_svm_classifier(train_path, test_path, None, True, kernel="poly")
    pipeline.evaluate_metrics(class_0, class_1, y_test, y_pred, y_proba)


if __name__ == "__main__":
    run()
