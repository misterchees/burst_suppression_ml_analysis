"""Contains the SplitManager class."""
import pandas as pd
from MachineLearning.IO.io_core import IOCore
from MachineLearning.Utils.split_utils import SplitUtils


class SplitManager:
    """
    Handles stratified, subject-wise splitting of EEG feature datasets
    into train and test sets, ensuring no ResultID overlap.
    """
    io_core = IOCore()

    def __init__(self, parameters: dict, class_0: str, class_1: str, test_size: float = 0.2, random_state: int = 42):
        """
        Initializes the SplitManager class, setting the paths of class 0 and class 1.
        :param parameters: Current parameters of the Epochs, from which the feature sets were calculated.
        :param class_0: Should be different from class_1. Valid options are: 'awake', 'faw' and 'normal_an'
        :param class_1: Should be different from class_0. Valid options are: 'awake', 'faw' and 'normal_an'
        :param test_size: Ratio of the test set, that define the test-train-split ratio.
        Valid options are between 0 and 1.
        :param random_state: Random state for splitting data into train and test sets (for reproducibility).
        """
        self.parameters = parameters
        self.class_1_path = self.io_core.return_file_fullpath(parameters, True, False, class_1,
                                                              "test_and_train_data", "feature_sets")
        self.class_0_path = self.io_core.return_file_fullpath(parameters, True, False, class_0,
                                                              "test_and_train_data", "feature_sets")
        self.test_size = test_size
        self.random_state = random_state
        self.class_0 = class_0
        self.class_1 = class_1

        self.class_1_df = None
        self.class_0_df = None
        self.train_df = None
        self.test_df = None

    def load_and_validate(self):
        """Loads both csv Files, each of them containing data of one class."""
        print(f"Loading {self.class_1} data from {self.class_1_path}\n "
              f"Loading {self.class_0} data from {self.class_0_path}")
        class_1_df = pd.read_csv(self.class_1_path)
        class_0_df = pd.read_csv(self.class_0_path)

        if list(class_1_df.columns) != list(class_0_df.columns):
            raise ValueError("Feature CSVs do not have the same columns.")

        self.class_1_df = class_1_df
        self.class_0_df = class_0_df
        print("Loading successful")

    def create_split(self):
        """
         Creates splits in the following fashion:
         1. Split on patient level in a way, that samples (epochs) are close to test/train ratio.
         2. Balance both classes to ratio of 50/50 on sample level
         3. Ensure that test/train ratio is still present on sample level. Optionally Downsize set that surpasses ratio.
         4. Sorts rows by ascending order of "label", "ResultID", "Start" in both sets.
         5. Saves sets to this class. To save them to your folder use the save method.
        """

        print(f"Creating Splits. Ratio: ({(1 - self.test_size) * 100:.1f}/{self.test_size * 100:.1f})")

        # Step 1: Assign labels according to file
        class_1_df = self.class_1_df.copy()
        class_1_df["label"] = 1
        class_0_df = self.class_0_df.copy()
        class_0_df["label"] = 0

        # Step 2: Find best split on patient level to reach test/train ratio on sample level
        train_ids, test_ids = SplitUtils.find_patient_split_by_epoch_balance(
            awake_df=self.class_1_df,
            faw_df=self.class_0_df,
            test_size=self.test_size,
            tolerance=0.05,
            random_state=self.random_state
        )

        # Step 3: Split Epochs along Patient IDs
        train_df = pd.concat([
            class_1_df[class_1_df["ResultID"].isin(train_ids)],
            class_0_df[class_0_df["ResultID"].isin(train_ids)]
        ], ignore_index=True)

        test_df = pd.concat([
            class_1_df[class_1_df["ResultID"].isin(test_ids)],
            class_0_df[class_0_df["ResultID"].isin(test_ids)]
        ], ignore_index=True)

        # Step 4: Balance both classes in test and train to 50/50
        test_df_balanced = SplitUtils.balance_classes(test_df, self.random_state)
        train_df_balanced = SplitUtils.balance_classes(train_df, self.random_state)

        # Step 5: Ensure that sample split has the correct ratio by adjusting to smaller set
        test_df_balanced, train_df_balanced = SplitUtils.adjust_splits_to_ratio(test_df_balanced, train_df_balanced,
                                                                                self.test_size, self.random_state)

        # Step 6: Sort rows
        self.train_df = train_df_balanced.sort_values(by=["label", "ResultID", "Start"]).reset_index(drop=True)
        self.test_df = test_df_balanced.sort_values(by=["label", "ResultID", "Start"]).reset_index(drop=True)

        print("Split creation successful.")
        print(f"Train set size: {len(self.train_df)}, Test set size: {len(self.test_df)}")
        print(f"Class distribution in train: {self.class_1}={sum(self.train_df.label == 1)},"
              f"{self.class_0}={sum(self.train_df.label == 0)}")
        print(f"Class distribution in test:  {self.class_1}={sum(self.test_df.label == 1)},"
              f" {self.class_0}={sum(self.test_df.label == 0)}")

    def save(self):
        """Saves created train and test splits to a folder defined by current parameters."""
        train_fullpath = self.io_core.return_split_folder_fullpath(self.parameters, "train")
        test_fullpath = self.io_core.return_split_folder_fullpath(self.parameters, "test")

        print(f"Saving train data to {train_fullpath}\n Saving test data to {test_fullpath}")
        self.train_df.to_csv(train_fullpath, index=False)
        self.test_df.to_csv(test_fullpath, index=False)

        print("Split saving successful")

    def return_split_paths(self):
        """Returns a tuple of paths to the train-test-split. Tuple -> (train_path, test_path)"""
        train_fullpath = self.io_core.return_split_folder_fullpath(self.parameters, "train")
        test_fullpath = self.io_core.return_split_folder_fullpath(self.parameters, "test")
        return train_fullpath, test_fullpath
