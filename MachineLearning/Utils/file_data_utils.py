from pathlib import Path
import pandas as pd
import numpy as np

class FileDataUtils:
    @staticmethod
    def serialize_for_json(obj):
        """
        Converts a given object into a JSON-compatible version.
        Supports pandas.DataFrame and numpy.ndarray.

        :param obj: Any object (dict, list, DataFrame, ndarray, ...)
        :return: JSON-compatible version of the object.
        """
        if isinstance(obj, pd.DataFrame):
            return {
                "__type__": "DataFrame",
                "data": obj.values.tolist(),
                "index": obj.index.tolist(),
                "columns": obj.columns.tolist()
            }
        elif isinstance(obj, np.ndarray):
            return {
                "__type__": "ndarray",
                "data": obj.tolist(),
                "shape": obj.shape
            }
        elif isinstance(obj, dict):
            return {k: FileDataUtils.serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [FileDataUtils.serialize_for_json(item) for item in obj]
        else:
            return obj

    @staticmethod
    def deserialize_from_json(obj):
        """
        Deserialize objects, which were serialized by serialize_for_json.
        Supports pandas.DataFrame und numpy.ndarray.

        :param obj: JSON-compatible object (dict, list, ...)
        :return: Deserialized original (DataFrame, ndarray, ...)
        """
        if isinstance(obj, dict):
            obj_type = obj.get("__type__")

            if obj_type == "DataFrame":
                return pd.DataFrame(
                    data=obj["data"],
                    index=obj["index"],
                    columns=obj["columns"]
                )
            elif obj_type == "ndarray":
                return np.array(obj["data"]).reshape(obj["shape"])
            else:
                # Recursively deserialize
                return {k: FileDataUtils.deserialize_from_json(v) for k, v in obj.items()}

        elif isinstance(obj, list):
            return [FileDataUtils.deserialize_from_json(item) for item in obj]

        else:
            return obj

    @staticmethod
    def append_unique_rows_to_csv(df: pd.DataFrame, csv_path: Path) -> tuple[pd.DataFrame, int]:
        """
        Appends a DataFrame to a CSV file, ensuring no duplicate rows are written.
        If the CSV does not exist, it will be created.

        Assumes that all rows in the DataFrame and CSV have the same structure (i.e., same columns).

        :param df: DataFrame to append.
        :param csv_path: Path to the target CSV file.
        :return: A tuple containing the DataFrame with the new rows and the number of new rows added.
        """

        if csv_path.exists():
            # Load path if exists
            existing_df = pd.read_csv(csv_path)
            # Append new rows that are not duplicates
            combined_df = pd.concat([existing_df, df], ignore_index=True).drop_duplicates()
            # Track new rows
            new_rows_df = pd.concat([combined_df, existing_df]).drop_duplicates(keep=False)
        else:
            combined_df = df.drop_duplicates(ignore_index=True)
            new_rows_df = combined_df.copy()

        # Overwrite the old file with updated df
        combined_df.to_csv(csv_path, index=False)

        return new_rows_df, len(new_rows_df)