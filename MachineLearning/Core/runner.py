"""
This module is to execute all pipeline commands and therefore the main point to run the project.
"""
from MachineLearning.Core.pipeline import Pipeline

INITIAL_DATA_SUBDIR_KEY = "combined_raw_data"


def run():
    """
    Function to execute any code
    """
    # Initialize with key of directory with patient ID subset of interest. Faw and awake flags are default true
    pipeline = Pipeline(INITIAL_DATA_SUBDIR_KEY)
    # pipeline.raw_eeg_filtering()
    # pipeline.transform_eeg_to_psd()
    # pipeline.feature_extraction(False, "mean", "variance", "bandpower", "spectral_skewness",
    #                            "spectral_kurtosis", "shannon_entropy")
    # pipeline.combine_all_features()
    pipeline.combine_features("mean", "variance", "bandpower", "spectral_skewness",
                              "spectral_kurtosis", "shannon_entropy")
    pipeline.create_splits(0.15, 42)
    # pipeline.run_svm_classifier()


if __name__ == "__main__":
    run()
