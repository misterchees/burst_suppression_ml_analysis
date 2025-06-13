import pandas as pd
from MachineLearning.IO.io_core import IOCore


class SplitManager:
    """
    Handles stratified, subject-wise splitting of EEG feature datasets
    into train and test sets, ensuring no ResultID overlap.
    """
    io_core = IOCore()

    def __init__(self, parameters: dict, test_size: float = 0.2, random_state: int = 42):
        self.parameters = parameters
        self.awake_path = self.io_core.return_awake_file_fullpath(parameters, "test_and_train_data", "feature_sets")
        self.faw_path = self.io_core.return_faw_file_fullpath(parameters, "test_and_train_data", "feature_sets", False)
        self.test_size = test_size
        self.random_state = random_state

        self.awake_df = None
        self.faw_df = None
        self.train_df = None
        self.test_df = None

    def load_and_validate(self):
        print(f"Loading awake data from {self.awake_path}\n Loading faw data from {self.faw_path}")
        awake_df = pd.read_csv(self.awake_path)
        faw_df = pd.read_csv(self.faw_path)

        if list(awake_df.columns) != list(faw_df.columns):
            raise ValueError("Feature CSVs do not have the same columns.")

        self.awake_df = awake_df
        self.faw_df = faw_df
        print("Loading successful")

    def create_split(self):
        from sklearn.model_selection import train_test_split
        import numpy as np

        print(f"Creating Splits. Ratio: ({(1-self.test_size)*100}/{self.test_size*100})")
        awake_ids = self.awake_df['ResultID'].unique()
        faw_ids = self.faw_df['ResultID'].unique()

        all_ids = np.concatenate([awake_ids, faw_ids])
        labels = np.concatenate([np.ones_like(awake_ids), np.zeros_like(faw_ids)])

        train_ids, test_ids, _, _ = train_test_split(
            all_ids, labels, test_size=self.test_size, stratify=labels, random_state=self.random_state
        )

        # Add label columns
        awake_train = self.awake_df[self.awake_df['ResultID'].isin(train_ids)].copy()
        awake_train['label'] = 1

        faw_train = self.faw_df[self.faw_df['ResultID'].isin(train_ids)].copy()
        faw_train['label'] = 0

        awake_test = self.awake_df[self.awake_df['ResultID'].isin(test_ids)].copy()
        awake_test['label'] = 1

        faw_test = self.faw_df[self.faw_df['ResultID'].isin(test_ids)].copy()
        faw_test['label'] = 0

        # Concat all
        self.train_df = pd.concat([awake_train, faw_train], ignore_index=True)
        self.test_df = pd.concat([awake_test, faw_test], ignore_index=True)

        print("Split creation successful")

    def save(self):
        train_fullpath = self.io_core.return_split_folder_fullpath(self.parameters, "train")
        test_fullpath = self.io_core.return_split_folder_fullpath(self.parameters, "test")

        print(f"Saving train data to {train_fullpath}\n Saving test data to {test_fullpath}")
        self.train_df.to_csv(train_fullpath, index=False)
        self.test_df.to_csv(test_fullpath, index=False)

        print("Split saving successful")

    def return_split_paths(self):
        train_fullpath = self.io_core.return_split_folder_fullpath(self.parameters, "train")
        test_fullpath = self.io_core.return_split_folder_fullpath(self.parameters, "test")
        return train_fullpath, test_fullpath
