"""This module contains the RunMetadata class."""
from datetime import datetime
from typing import List, Union, Optional, Dict, Any
from pathlib import Path

import pandas as pd

from MachineLearning.IO.save_result import SaveResult
from MachineLearning.IO.load_data import LoadData
from MachineLearning.Utils.path_manager import PathManager
from MachineLearning.Utils.path_utils import PathUtils
from MachineLearning.Utils.config_handler import load_config, update_config


class RunMetadata:
    """
    Container for collecting and managing metadata for a single ML run.
    Allows step-by-step population of fields and final export to YAML.
    """
    # Keys that are used multiple times in this class
    epoch_types_key = "epoch_type"
    model_params_key = "model"
    initial_patient_ids_key = "initial_patient_ids"
    hyperparameters_key = "hyperparameters"
    filtering_params_key = "filtering_params"
    transform_params_key = "transform_params"
    feature_params_key = "feature_params"
    classification_params_key = "classification_params"


    def __init__(
            self,
            epoch_types: list,
            model_params: dict,
            initial_patient_ids: Union[List[int], set],
            hyperparameters: dict,
            filtering_params: dict,
            normalize_method: str,
            transform_params: dict,
            classification_params: dict,
            run_name: Optional[str] = None,
            force_overwrite: bool = False
    ):
        """
        :param epoch_types: List of exactly 2 epoch types to classify. Valid options are "normal_an", "awake", "faw"
        :param model_params: Params for the model used in this run.
        :param initial_patient_ids: List or set of patient IDs used in this run
        :param hyperparameters: Dictionary of hyperparameters used in this run
        :param filtering_params: Dictionary of filtering parameters used in this run.
        :param normalize_method: Method used for normalization.
        :param transform_params: Dictionary of transform parameters used in this run.
        :param classification_params: Dictionary of classification parameters used in this run.
        :param run_name: Optional manual run name (e.g. timestamp or hash)
        :param force_overwrite: If False, cancels run if there is already one with the same parameters.
        """
        for epoch_type in epoch_types:
            if epoch_type not in ["normal_an", "awake", "faw"]:
                raise ValueError(f"Invalid epoch_type: {epoch_type}")

        # Initial params
        self.epoch_types = epoch_types
        self.model_key = next(iter(model_params))
        self.model_dict = model_params
        self.initial_patient_ids = sorted(list(initial_patient_ids))
        self.hyperparameters = hyperparameters
        self.filtering_params = filtering_params
        self.normalize_method = normalize_method
        self.transform_params = transform_params
        self.classification_params = classification_params
        self.param_hash = self._calculate_dict_hash(epoch_types, model_params, initial_patient_ids, hyperparameters,
                                                    filtering_params, transform_params, classification_params)

        # Cancel run, if there is already one existent
        if not force_overwrite:
            path_of_exact_same_run = self._find_run_with_same_parameters()
            if path_of_exact_same_run:
                raise ValueError(f"A run with the same parameters already exists at {path_of_exact_same_run}.")

        # Params that are collected in and after the pipeline run
        self.final_patient_ids = set()
        self.feature_params = {}
        self.split_data = {}
        self.metrics = None
        self.additional_info = {}

        # Set run_name and timestamp for this instance
        self.run_name, self.timestamp = self._init_run_name(run_name)
        # Set run_name globally
        update_config("parameters_config.yaml", {"run_name": self.run_name})


    def set_feature_info(self):
        """Store all feature params based on the given list of used features."""
        featureset_params = load_config("parameters_config.yaml")["feature_params"]
        rel_bandpower_key = "relative_bandpower"

        # Assemble path and read header of combined features csv
        pm = PathManager()
        combined_features_path = pm.resolve_episode_path(
            self.hyperparameters, self.epoch_types[0], ["test_and_train_data", "feature_sets"], True, False
        )
        combined_features_df_header = pd.read_csv(combined_features_path, nrows=0)

        # Change to list and remove non-feature columns as well as single bands
        header_list = combined_features_df_header.columns.to_list()
        bands_to_remove = list(featureset_params[rel_bandpower_key]["frequency_bands"].keys())
        feature_list = [col for col in header_list if col not in {"Start", "End", "ResultID"} and col not in bands_to_remove]

        # Add "relative_bandpower" as a replacement if single bands were present
        if any(col in header_list for col in bands_to_remove):
            feature_list.append(rel_bandpower_key)

        # Get all relevant features with params from all params dict
        for feature in feature_list:
            feature = feature.lower()  # Features to lower case, to match with param keys
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
            fold_number = str(Path(train_path).stem).split("_")[0] # Last file of Path is the stem
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

    def set_final_result_ids(self, folds_dict: dict):
        """Store ResultIDs that were finally used for the classification."""
        self.final_patient_ids.update(folds_dict["ResultID"])

    def set_metrics(self, metrics: dict):
        """Store evaluation metrics."""
        self.metrics = metrics

    def add_info(self, key: str, value: Any):
        """Generic setter for any extra metadata."""
        self.additional_info[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Returns the entire metadata as a serializable dictionary."""
        return {
            "run_name": self.run_name,
            "timestamp": self.timestamp,
            self.epoch_types_key: self.epoch_types,
            self.model_params_key: self.model_dict,
            self.initial_patient_ids_key: self.initial_patient_ids,
            "final_patient_ids": sorted(self.final_patient_ids),
            self.hyperparameters_key: self.hyperparameters,
            self.filtering_params_key: self.filtering_params,
            "normalize_method": self.normalize_method,
            self.transform_params_key: self.transform_params,
            self.feature_params_key: self.feature_params,
            "split_data": self.split_data,
            self.classification_params_key: self.classification_params,
            "metrics": self.metrics,
            "additional_info": self.additional_info,
            "param_hash": self.param_hash
        }

    def save_to_json(self):
        """Save the metadata to a JSON file."""
        saver = SaveResult()
        saver.save_run_metadata_to_json(self.hyperparameters, self.model_key, self.to_dict(), Path(f"{self.run_name}.json"))

    def __repr__(self):
        """String representation of the RunMetadata object."""
        return f"<RunMetadata run='{self.run_name}' model='{self.model_key}'>"

    def _init_run_name(self, run_name: str | None) -> tuple[str, str]:
        """
        Initializes the run name, ensuring uniqueness within the metadata directory.

        :param run_name: Optional base name for the run.
        :return: Unique run name as string.
        """
        timestamp = datetime.now()
        metadata_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")  # Timestamp for the metadata dict

        timestamp_suffix = timestamp.strftime("%Y_%m_%dT%H_%M_%S")  # Timestamp suffix for the run_name
        default_run_name = f"Run_{timestamp_suffix}"

        base_name = run_name or default_run_name

        # Get folder with files and return a list of them
        pm = PathManager()
        metadata_dir = pm.get_complex_ml_path(
            self.hyperparameters, ["run_metadata", self.model_key], False, False
        )
        files,_ = PathUtils.list_files_in_folder(metadata_dir, ".json")

        # Files list to set and then check if run_name already exists
        existing_runs = set(files) if files else set()

        if f"{base_name}.json" in existing_runs:
            base_name = f"{base_name}_{timestamp_suffix}"  # Append timestamp if name already exists

        return base_name, metadata_timestamp

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

    def _find_run_with_same_parameters(self) -> Path | None:
        """
        Checks for the existence of a specific parameter hash in previously stored metadata
        files within a given model's metadata directory. If the hash exists, the path to
        the corresponding metadata file is returned.

        This method iterates over metadata files associated with the current model. It
        verifies whether the `param_hash` matches the stored hash in the metadata. If
        found, it returns the path to the matching file. If no associated hash exists
        within the metadata structure, a new hash is added for existing runs and checked
        against the current parameter hash.

        :rtype: Path or None
        :return: The path to the metadata file containing the matching parameter hash if
            found, otherwise returns None.
        """

        # Get folder with files and return a list of them
        pm = PathManager()
        metadata_dir = pm.get_complex_ml_path(
            self.hyperparameters, ["run_metadata", self.model_key], False, False
        )
        files, _ = PathUtils.list_files_in_folder(metadata_dir, ".json")

        for filename in files:
            # Load metadata file
            fullpath = Path(metadata_dir, filename)
            metadata_dict = LoadData.load_json(fullpath)

            # Checks if there is a hash present in old runs. If not, creates a new one and updates the existing run
            if "param_hash" in metadata_dict:
                if metadata_dict["param_hash"] == self.param_hash:
                    return fullpath
            else:
                other_hash = self._add_hash_to_existing_run(metadata_dict, filename)
                if other_hash == self.param_hash:
                    return fullpath

        return None


    def _add_hash_to_existing_run(self, metadata_dict: dict, file: Path) -> str:
        """
        Adds a hash based on provided metadata to an existing run and updates the metadata file.

        The method calculates a hash from specific metadata fields and appends it to the given
        metadata dictionary before saving it to a file. The hash is returned after being added to
        the dictionary.

        :param metadata_dict: Metadata dictionary containing information used to generate the hash.
        :param file: File path where the metadata, along with the generated hash, will be saved.
        :return: The hash string generated from the provided metadata.
        """
        other_hash = self._calculate_dict_hash(
            epoch_types=metadata_dict[self.epoch_types_key],
            model_params=metadata_dict[self.model_params_key],
            initial_patient_ids=metadata_dict[self.initial_patient_ids_key],
            hyperparameters=metadata_dict[self.hyperparameters_key],
            filtering_params=metadata_dict[self.filtering_params_key],
            transform_params=metadata_dict[self.transform_params_key],
            classification_params=metadata_dict[self.classification_params_key]
        )
        metadata_dict["param_hash"] = other_hash
        saver = SaveResult()
        saver.save_run_metadata_to_json(self.hyperparameters, self.model_key, metadata_dict,
                                        file)
        return other_hash

    def _calculate_dict_hash(
            self,
            epoch_types: list,
            model_params: dict,
            initial_patient_ids: Union[List[int], set],
            hyperparameters: dict,
            filtering_params: dict,
            transform_params: dict,
            classification_params: dict,) -> str:
        """
        Generates a unique hash for the given parameters. For more information about the params, see: __init__
        :return: SHA256 hash string.
        """
        import json
        import hashlib
        param_fingerprint_dict = {
            self.epoch_types_key: sorted(epoch_types),
            self.model_params_key: model_params,
            self.initial_patient_ids_key: sorted(list(initial_patient_ids)),
            self.hyperparameters_key: hyperparameters,
            self.filtering_params_key: filtering_params,
            self.transform_params_key: transform_params,
            self.classification_params_key: classification_params
        }

        # Convert to stable JSON string (sort keys, remove whitespace)
        param_str = json.dumps(param_fingerprint_dict, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(param_str.encode('utf-8')).hexdigest()
