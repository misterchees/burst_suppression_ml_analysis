import pandas as pd
from MachineLearning.IO.io_core import IOCore


class SplitManager:
    """
    Handles stratified, subject-wise splitting of EEG feature datasets
    into train and test sets, ensuring no ResultID overlap.
    """
    io_core = IOCore()

    def __init__(self, parameters: dict, test_size: float = 0.2, random_state: int = 42, normal_an=False):
        self.parameters = parameters
        self.awake_path = self.io_core.return_awake_file_fullpath(parameters, True, "test_and_train_data", "feature_sets")
        if normal_an:
            self.faw_path = self.io_core.return_normal_an_file_fullpath(parameters,True, "test_and_train_data", "feature_sets")
        else:
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
        import numpy as np
        from sklearn.model_selection import train_test_split

        print(f"Creating patient-level balanced sample splits with ratio: "
              f"({(1 - self.test_size) * 100:.0f}/{self.test_size * 100:.0f})")

        # Step 1: Unique patient IDs
        awake_ids = self.awake_df['ResultID'].unique()
        faw_ids = self.faw_df['ResultID'].unique()

        # Labels for stratified splitting
        all_ids = np.concatenate([awake_ids, faw_ids])  # Array for ResultIDs
        labels = np.concatenate([np.ones_like(awake_ids), np.zeros_like(faw_ids)])  # ResultIDs mapped to labels [0,1]

        # Stratify to preserve ratio in test and train and split along labels (resultID in test -> not in train)
        train_ids, test_ids, _, _ = train_test_split(
            all_ids, labels, test_size=self.test_size, stratify=labels, random_state=self.random_state
        )

        # Select patients for each set
        awake_train = self.awake_df[self.awake_df['ResultID'].isin(train_ids)].copy()
        faw_train = self.faw_df[self.faw_df['ResultID'].isin(train_ids)].copy()

        awake_test = self.awake_df[self.awake_df['ResultID'].isin(test_ids)].copy()
        faw_test = self.faw_df[self.faw_df['ResultID'].isin(test_ids)].copy()

        # Step 2: Balance each set by sample count (truncate to the smallest class)
        def balance_samples(df1, df2, label1, label2):
            df1 = df1.copy()
            df2 = df2.copy()
            df1['label'] = label1
            df2['label'] = label2

            min_len = min(len(df1), len(df2))
            df1 = df1.sample(n=min_len, random_state=self.random_state)
            df2 = df2.sample(n=min_len, random_state=self.random_state)

            return pd.concat([df1, df2], ignore_index=True).sample(frac=1, random_state=self.random_state).reset_index(
                drop=True)

        self.train_df = balance_samples(awake_train, faw_train, 1, 0)
        self.test_df = balance_samples(awake_test, faw_test, 1, 0)

        # Sort splits
        self.train_df = self.train_df.sort_values(by=["label", "ResultID", "Start"], ascending=[False, True, True])
        self.test_df = self.test_df.sort_values(by=["label", "ResultID", "Start"], ascending=[False, True, True])

        print(f"Split creation successful.")
        print(f"Train set: {len(self.train_df)} samples, Test set: {len(self.test_df)} samples")
        print(
            f"Class distribution in train: awake={sum(self.train_df.label == 1)}, faw={sum(self.train_df.label == 0)}")
        print(f"Class distribution in test:  awake={sum(self.test_df.label == 1)}, faw={sum(self.test_df.label == 0)}")

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
