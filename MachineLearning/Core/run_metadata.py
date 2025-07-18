"""This module contains the RunMetadata class."""
import os
import yaml
from datetime import datetime
from typing import List, Union, Optional, Dict, Any


class RunMetadata:
    """
    Container for collecting and managing metadata for a single ML run.
    Allows step-by-step population of fields and final export to YAML.
    """

    def __init__(
            self,
            epoch_types: list,
            model_key: str,
            initial_patient_ids: Union[List[int], set],
            hyperparameters: dict,
            filtering_params: dict,
            run_name: Optional[str] = None
    ):
        """
        :param epoch_types: List of exactly 2 epoch types to classify. Valid options are "normal_an", "awake", "faw"
        :param model_key: String identifier for the ML model (e.g. "svm")
        :param initial_patient_ids: List or set of patient IDs used in this run
        :param hyperparameters: Dictionary of hyperparameters used in this run
        :param filtering_params: Dictionary of filtering parameters used in this run.
        :param run_name: Optional manual run name (e.g. timestamp or hash)
        """
        if epoch_types not in {"normal_an", "awake", "faw"}:
            raise ValueError(f"Invalid epoch_type: {epoch_types}")

        self.epoch_types = epoch_types
        self.model_key = model_key
        self.initial_patient_ids = sorted(list(initial_patient_ids))
        self.hyperparameters = hyperparameters
        self.filtering_params = filtering_params
        self.additional_info = {}

        # Timestamp as the default Run-Name and as meta-information
        timestamp = datetime.now()
        self.timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        self.run_name = run_name or timestamp.strftime("%Y_%m_%dT%H_%M_%S")

    def set_final_result_ids(self, result_ids: List[int]):
        """Store ResultIDs that were finally used for the classification."""
        self.additional_info["final_patient_ids"] = sorted(result_ids)

    def set_split_file(self, filename: str):
        """Save the split filename that was used (e.g. for train/test split)."""
        self.additional_info["split_file"] = filename

    def set_metrics(self, metrics: Dict[str, float]):
        """Store evaluation metrics."""
        self.additional_info["metrics"] = metrics

    def set_feature_info(self, features_used: List[str]):
        """Store list of features used."""
        self.additional_info["features_used"] = features_used

    def add_info(self, key: str, value: Any):
        """Generic setter for any extra metadata."""
        self.additional_info[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Returns the entire metadata as a serializable dictionary."""
        return {
            "run_name": self.run_name,
            "timestamp": self.timestamp,
            "epoch_type": self.epoch_types,
            "model_key": self.model_key,
            "patient_ids": self.initial_patient_ids,
            **self.additional_info
        }

    def save_to_yaml(self, output_dir: str = "Run_informations"):
        """Save the metadata to a YAML file in the given directory."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{self.run_name}.yaml")
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def __repr__(self):
        return f"<RunMetadata run='{self.run_name}' model='{self.model_key}'>"
