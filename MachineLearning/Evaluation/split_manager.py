"""Contains the SplitManager class."""
import numpy as np
import pandas as pd

from MachineLearning.IO.io_core import IOCore, PathUtils
from MachineLearning.IO.save_result import SaveResult
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
        self.k_folds = None

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

    def create_single_split(self, save=True):
        """
         Creates splits in the following fashion:
         1. Split on patient level in a way, that samples (epochs) are close to test/train ratio.
         2. Balance both classes to ratio of 50/50 on sample level
         3. Ensure that test/train ratio is still present on sample level. Optionally Downsize set that surpasses ratio.
         4. Sorts rows by ascending order of "label", "ResultID", "Start" in both sets.
         5. Saves sets to this class. To save them to your folder use the save method.

         :param save: Boolean flag to save the splits in csv files.
         :return: A tuple of the split dataframes in this order (train/test)
        """

        print(f"Creating Split. Ratio: ({(1 - self.test_size) * 100:.1f}/{self.test_size * 100:.1f})")

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

        # Step 3: Split along patient IDs, balance classes. (optional) Ensure ratio afterward.
        train_df, test_df = SplitUtils.return_balanced_split(
            train_ids=train_ids,
            test_ids=test_ids,
            random_state=self.random_state,
            class_0_df=class_0_df,
            class_1_df=class_1_df,
            test_size=self.test_size
        )

        # Step 4: Sort rows
        self.train_df = train_df.sort_values(by=["label", "ResultID", "Start"]).reset_index(drop=True)
        self.test_df = test_df.sort_values(by=["label", "ResultID", "Start"]).reset_index(drop=True)

        # Save if requested
        if save:
            test_train = (self.train_df, self.test_df)
            self.save(test_train)

        print("Split creation successful.")
        print(f"Train set size: {len(self.train_df)}, Test set size: {len(self.test_df)}")
        print(f"Class distribution in train: {self.class_1}={sum(self.train_df.label == 1)},"
              f"{self.class_0}={sum(self.train_df.label == 0)}")
        print(f"Class distribution in test:  {self.class_1}={sum(self.test_df.label == 1)},"
              f" {self.class_0}={sum(self.test_df.label == 0)}")

        return self.train_df, self.test_df

    def create_cv_splits(self, n_splits=5, save=True):
        """
        Creates patient-based cross-validation splits for use in GridSearchCV.
        - Balances each split to have 50/50 class ratio
        - Returns feature matrix, label vector, and list of (train_idx, test_idx)

        :param n_splits: Number of CV folds
        :param save: Boolean flag to save the splits in csv files.
        :returns: (X, y, splits)
        """
        from sklearn.model_selection import KFold

        # Create set of present patient IDs
        all_ids = pd.concat([self.class_1_df, self.class_0_df])['ResultID'].unique()
        # create k-fold cross-validator
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        # Create full dataset
        full_df = SplitUtils.create_full_df(self.class_1_df, self.class_0_df)

        splits = []

        # Split sets with help of k-fold
        for train_id_idx, test_id_idx in kf.split(all_ids):
            train_ids = all_ids[train_id_idx]
            test_ids = all_ids[test_id_idx]

            train_idx, test_idx = SplitUtils.return_train_test_sample_idx(
                train_ids=train_ids,
                test_ids=test_ids,
                random_state=self.random_state,
                full_df=full_df,
                test_size=self.test_size
            )

            splits.append((train_idx, test_idx))  # Save split indices as Tuple -> (train, test)

        X, y = SplitUtils.return_X_y(full_df)
        self.k_folds = len(splits)  # Save number of folded splits created

        # Save if requested
        if save:
            split_obj = (X,y,splits)
            self.save(split_obj)

        return X, y, splits

    def create_custom_splits_by_test_size(self, save=True):
        splits = []
        non_overlap_split_number = int(1//self.test_size) # Maximum possible number of non overlapping splits
        print(f"Creating folded non overlapping Splits. "
              f"Ratio: ({(1 - self.test_size) * 100:.1f}/{self.test_size * 100:.1f})")

        full_df = SplitUtils.create_full_df(self.class_1_df, self.class_0_df)

        used_test_ids = set()
        split_counter = 0

        for _ in range(non_overlap_split_number):

            # Should only fail due to too small of a sample for current iteration
            try:
                train_ids, test_ids = SplitUtils.find_patient_split_by_epoch_balance(
                    awake_df=self.class_1_df,
                    faw_df=self.class_0_df,
                    test_size=self.test_size,
                    tolerance=0.05,
                    random_state=np.random.randint(10000),
                    exclude_ids=used_test_ids
                )
            except ValueError as e:
                print(f"Split creation failed for Split number: {split_counter}. Error is:\n{e}")
                break

            # Add found test_ids to avoid getting them in subsequent splits
            used_test_ids.update(test_ids)

            # Get indices of split
            train_idx, test_idx = SplitUtils.return_train_test_sample_idx(
                train_ids=train_ids,
                test_ids=test_ids,
                random_state=self.random_state,
                full_df=full_df,
                test_size=self.test_size
            )

            splits.append((train_idx, test_idx))
            split_counter += 1

        X, y = SplitUtils.return_X_y(full_df)
        self.k_folds = len(splits)  # Save number of folded splits created

        # Save if requested
        if save:
            split_obj = (X, y, splits)
            self.save(split_obj)

        return X, y, splits

    def save(self, split_obj):
        """Save single split or k-fold splits."""
        saver = SaveResult()
        # Check if three values in split_obj -> X, y, splits. If not, it must be split_obj -> train_df, test_df
        try:
            _, _, _ = split_obj
        except ValueError:
            saver.save_single_split(self.parameters, split_obj)
            print("Single split saving successful")
            return
        saver.save_cv_splits_to_csv(self.parameters, split_obj)
        print("CV splits saving successful")

    def return_split_paths(self):
        """Returns a tuple of paths to the train-test-split. Tuple -> (train_path, test_path)"""
        train_fullpath = self.io_core.return_single_split_folder_fullpath(self.parameters, "train")
        test_fullpath = self.io_core.return_single_split_folder_fullpath(self.parameters, "test")
        return train_fullpath, test_fullpath

    def return_k_fold_split_paths(self) -> list:
        """
        Returns a list of tuples containing the paths for the train and test folds.
        If one of them doesn't exist, a FileNotFoundError is raised.

        :return: List of tuples. Every tuple -> (train_path, test_path)
        """
        splits_list = []

        # alias functions
        exists = PathUtils.filepath_exists
        get_fullpath = self.io_core.return_folded_split_folder_fullpath

        for fold in range(self.k_folds):
            fold+=1  # folds start at 1 and not 0
            train_fullpath = get_fullpath(self.parameters, "train", fold, self.k_folds, False)
            test_fullpath = get_fullpath(self.parameters, "test", fold, self.k_folds, False)

            # Check if files exist
            if not exists(train_fullpath) or not exists(test_fullpath):
                raise FileNotFoundError(f"Could not find {train_fullpath} or {test_fullpath}")

            splits_list.append((train_fullpath, test_fullpath))

        return splits_list



