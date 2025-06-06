from MachineLearning.Core.pipeline import Pipeline

initial_data_subdir_key = "combined_raw_data"


def run():
    """
    Function to execute any code
    """
    # Initialize with directory key of directory with patient ID subset of interest
    pipeline = Pipeline(initial_data_subdir_key)
    pipeline.raw_eeg_filtering()


if __name__ == "__main__":
    run()
