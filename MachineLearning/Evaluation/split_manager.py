import pandas as pd
from MachineLearning.IO.io_core import IOCore


class SplitManager:
    """
    Handles stratified, subject-wise splitting of EEG feature datasets
    into train and test sets, ensuring no ResultID overlap.
    """
    io_core = IOCore()

    def __init__(self, parameters: dict, test_size: float = 0.2, random_state: int = 42):
        self.awake_path = self.io_core.return_awake_file_fullpath(parameters, "test_and_train_data", "feature_sets")
        self.faw_path = self.io_core.return_faw_file_fullpath(parameters, "test_and_train_data", "feature_sets", False)
        self.test_size = test_size
        self.random_state = random_state

        self.awake_df = None
        self.faw_df = None
        self.train_df = None
        self.test_df = None

    def load_and_validate(self):
        awake_df = pd.read_csv(self.awake_path)
        faw_df = pd.read_csv(self.faw_path)

        if list(awake_df.columns) != list(faw_df.columns):
            raise ValueError("Feature CSVs do not have the same columns.")

        self.awake_df = awake_df
        self.faw_df = faw_df

    def create_split(self):
        from sklearn.model_selection import train_test_split
        import numpy as np

        awake_ids = self.awake_df['ResultID'].unique()
        faw_ids = self.faw_df['ResultID'].unique()

        all_ids = np.concatenate([awake_ids, faw_ids])
        labels = np.concatenate([np.ones_like(awake_ids), np.zeros_like(faw_ids)])

        train_ids, test_ids, _, _ = train_test_split(
            all_ids, labels, test_size=self.test_size, stratify=labels, random_state=self.random_state
        )

        self.train_df = pd.concat([
            self.awake_df[self.awake_df['ResultID'].isin(train_ids)],
            self.faw_df[self.faw_df['ResultID'].isin(train_ids)]
        ], ignore_index=True)

        self.test_df = pd.concat([
            self.awake_df[self.awake_df['ResultID'].isin(test_ids)],
            self.faw_df[self.faw_df['ResultID'].isin(test_ids)]
        ], ignore_index=True)

    def save(self, output_dir: str):
        # Both in Test_and_train/Splits/<parameter defined folder>
        self.train_df.to_csv(f"{output_dir}/train_split.csv", index=False)
        self.test_df.to_csv(f"{output_dir}/test_split.csv", index=False)
