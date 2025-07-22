"""This module contains the RunMetadata class."""
from datetime import datetime
from typing import List, Union, Optional, Dict, Any

import pandas as pd

from MachineLearning.IO.save_result import SaveResult, PathUtils
from MachineLearning.Utils.config_handler import load_config


class RunMetadata:
    """
    Container for collecting and managing metadata for a single ML run.
    Allows step-by-step population of fields and final export to YAML.
    """

    def __init__(
            self,
            epoch_types: list,
            model_params: dict,
            initial_patient_ids: Union[List[int], set],
            hyperparameters: dict,
            filtering_params: dict,
            transform_params: dict,
            run_name: Optional[str] = None
    ):
        """
        :param epoch_types: List of exactly 2 epoch types to classify. Valid options are "normal_an", "awake", "faw"
        :param model_params: Params for the model used in this run.
        :param initial_patient_ids: List or set of patient IDs used in this run
        :param hyperparameters: Dictionary of hyperparameters used in this run
        :param filtering_params: Dictionary of filtering parameters used in this run.
        :param transform_params: Dictionary of transform parameters used in this run.
        :param run_name: Optional manual run name (e.g. timestamp or hash)
        """
        if epoch_types not in {"normal_an", "awake", "faw"}:
            raise ValueError(f"Invalid epoch_type: {epoch_types}")

        # Initial params
        self.epoch_types = epoch_types
        self.model_key = next(iter(model_params))
        self.model_dict = model_params
        self.initial_patient_ids = sorted(list(initial_patient_ids))
        self.hyperparameters = hyperparameters
        self.filtering_params = filtering_params
        self.transform_params = transform_params

        # Params that are collected in and after the pipeline
        self.final_patient_ids = set()
        self.feature_params = {}
        self.split_data = {}
        self.classification_params = None
        self.metrics = None
        self.meta_analysis = None
        self.additional_info = {}

        # Timestamp as the default Run-Name and as meta-information
        timestamp = datetime.now()
        self.timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        self.run_name = run_name or timestamp.strftime("%Y_%m_%dT%H_%M_%S")

    def set_feature_info(self, feature_list: list):
        """Store all feature params based on the given list of used features."""
        featureset_params = load_config("parameters_config.yaml")["feature_params"]

        for feature in feature_list:
            feature_params = featureset_params[feature]
            self.feature_params[feature] = feature_params

    def set_split_data(self, split_paths: list):
        """
        Save the split data as a dictionary, using given split paths to retrieve the correct splits.
        Stucture in the end is:
        split_data: {
            fold_<fold number>: {
                test: {
                    Start: <List of starts>
                    End: <List of ends>
                    ResultID: <List of resultIDs>
                }
                train: {
                    Start: <List of starts>
                    End: <List of ends>
                    ResultID: <List of resultIDs>
                }
            }
        }

        :param split_paths: List of split paths to retrieve the correct splits.
        """

        for train_path, test_path in split_paths:

            # It doesn't matter from which path the fold number is retrieved (same for both)
            fold_number = PathUtils.return_filename_from_fullpath(train_path).split("_")[0]
            fold_key = f"fold_{fold_number}"

            # Create dicts for the single instances of the splits
            test_dict = self._create_split_subdict(test_path)
            train_dict = self._create_split_subdict(train_path)

            # Append to split data
            self.split_data[fold_key] = {
                "test": test_dict,
                "train": train_dict
                }

            # Update final patient IDs
            self.set_final_result_ids(test_dict)
            self.set_final_result_ids(train_dict)

    @staticmethod
    def _create_split_subdict(split_path: str):
        """Creates entry for split path with Lists of Start, End, ResultID in the same order."""

        fold_df = pd.read_csv(split_path)
        starts = fold_df["Start"].to_list()  # Order is preserved in this operation
        ends = fold_df["End"].to_list()
        result_ids = fold_df["ResultID"].to_list()

        subdict = {
            "Start": starts,
            "End": ends,
            "ResultID": result_ids
        }

        return subdict

    def set_final_result_ids(self, folds_dict: dict):
        """Store ResultIDs that were finally used for the classification."""
        self.final_patient_ids.update(folds_dict["ResultID"])

    def set_classification_params(self, classification_params: dict):
        """Store all classification params."""
        self.classification_params = classification_params

    def set_metrics(self, metrics: dict):
        """Store evaluation metrics."""
        self.metrics = metrics

    def set_meta_analysis(self, meta_analysis: dict):
        """Store meta analysis."""
        self.meta_analysis = meta_analysis

    def add_info(self, key: str, value: Any):
        """Generic setter for any extra metadata."""
        self.additional_info[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Returns the entire metadata as a serializable dictionary."""
        return {
            "run_name": self.run_name,
            "timestamp": self.timestamp,
            "epoch_type": self.epoch_types,
            "model": self.model_dict,
            "initial_patient_ids": self.initial_patient_ids,
            "final_patient_ids": sorted(self.final_patient_ids),
            "hyperparameters": self.hyperparameters,
            "filtering_params": self.filtering_params,
            "transform_params": self.transform_params,
            "feature_params": self.feature_params,
            "split_data": self.split_data,
            "classification_params": self.classification_params,
            "metrics": self.metrics,
            "meta_analysis": self.meta_analysis,
            "additional_info": self.additional_info
        }

    def save_to_json(self):
        """Save the metadata to a JSON file."""
        saver = SaveResult()
        saver.save_run_metadata_to_json(self.hyperparameters, self.model_key, self.to_dict(), f"{self.run_name}.json")

    def __repr__(self):
        """String representation of the RunMetadata object."""
        return f"<RunMetadata run='{self.run_name}' model='{self.model_key}'>"
