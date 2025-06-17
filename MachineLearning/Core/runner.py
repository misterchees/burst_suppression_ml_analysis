from MachineLearning.Core.pipeline import Pipeline

initial_data_subdir_key = "combined_raw_data"


def run():
    """
    Function to execute any code
    """
    # Initialize with key of directory with patient ID subset of interest. Faw and awake flags are default true
    pipeline = Pipeline(initial_data_subdir_key, faw=False, normal_an=True)
    # pipeline.raw_eeg_filtering()
    # pipeline.transform_eeg_to_psd()
    # pipeline.feature_extraction(False, "mean", "variance", "bandpower", "spectral_skewness", "spectral_kurtosis", "shannon_entropy")
    # pipeline.combine_all_features()
    pipeline.combine_features("mean", "variance", "bandpower", "shannon_entropy")
    pipeline.create_splits(0.15, 42, normal_an=True)
    # pipeline.run_svm_classifier()


if __name__ == "__main__":
    run()
