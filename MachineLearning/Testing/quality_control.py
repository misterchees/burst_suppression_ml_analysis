import pandas as pd

from MachineLearning.IO.save_result import SaveResult, PathUtils
from MachineLearning.Utils.config_loader import load_config


class QualityControl:
    def __init__(self):
        # Load initial params as params
        self.parameters = load_config("parameters_config.yaml")["initial_params"]
        pass

    def check_awake_and_faw_overlap(self, awake_path, faw_path):

        print("Checking awake and faw overlap")
        saver = SaveResult()

        # Get set of awake patient IDs
        awake_df = pd.read_csv(awake_path)
        awake_result_ids = list(awake_df["ResultID"].unique())

        # Load faw csv
        faw_df = pd.read_csv(faw_path)
        faw_result_ids = faw_df["ResultID"].unique()

        both_classes = []

        for result_id in awake_result_ids:
            if result_id in faw_result_ids:
                both_classes.append(1)
            else:
                both_classes.append(0)

        comparison_df = pd.DataFrame()
        comparison_df["awake_ids"] = awake_result_ids
        comparison_df["faw_exist"] = both_classes

        base_dir_path = saver.return_folder_path()
        file_name = "comparison_df.csv"
        fullpath = PathUtils.return_anypath(base_dir_path, file_name)
        PathUtils.save_file_as_csv(comparison_df, fullpath, False)

        print(f"Check succesful. Data saved to {fullpath}")

    def unique_faw_result_ids(self, faw_path):
        print("Saving unique result ids")
        saver = SaveResult()

        # Load faw csv
        faw_df = pd.read_csv(faw_path)
        faw_result_ids = pd.DataFrame(list(faw_df["ResultID"].unique()))

        base_dir_path = saver.return_folder_path()
        file_name = "unique_faw_result_ids.csv"
        fullpath = PathUtils.return_anypath(base_dir_path, file_name)
        PathUtils.save_file_as_csv(faw_result_ids, fullpath, False)





if __name__ == "__main__":
    awake_path = "D:\\Daten\\Test_and_train\\Feature_sets\\Awake_20.csv"
    faw_path = "D:\\Daten\\Test_and_train\\Feature_sets\\Feature_sets_70_080_20_5\\Summary_Episodes_20_000.csv"

    quality_control = QualityControl()
    quality_control.check_awake_and_faw_overlap(awake_path, faw_path)
    quality_control.unique_faw_result_ids(faw_path)
