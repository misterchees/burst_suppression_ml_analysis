import os

import pandas as pd
import numpy as np
import json
from pathlib import Path
from matplotlib import pyplot as plt


class PathUtils:
    @staticmethod
    def return_anypath(*path_parts: str) -> str:
        """
        Creates any path build from given path_parts.
        :param path_parts: Any number of path parts in the right order to assemble them as strings.
        :return: A path built from given path_parts as a string.
        """
        return str(os.path.join(*path_parts))

    @staticmethod
    def return_A_B_C_D_name(prefix, parameters: dict) -> str:
        """
        Calculates and returns a name variable defined by the class attributes,
        refractory time, min_episode_length, mac_threshold, and bis_threshold.
        Example: result_70_080_20_5

        :param prefix: Prefix of the subfolder name variable.
        :param parameters: A dictionary with all mentioned parameters

        :return: Subfolder string variable.
        """
        # extract parameters from dict
        mac_threshold = parameters["mac_threshold"]
        bis_threshold = parameters["bis_threshold"]
        min_episode_length = parameters["min_episode_length"]
        refractory_time = parameters["refractory_time"]

        # leave 2 digits after decimal point and remove it afterward: 0.5 -> 050, 0.25 -> 025 etc.
        mac_threshold = f"{mac_threshold:.2f}".replace(".", "")
        return f"{prefix}_{bis_threshold}_{mac_threshold}_{min_episode_length}_{refractory_time}"

    @staticmethod
    def return_X_Y_name(parameters: dict) -> str:
        """
        Calculates and returns a name variable defined by the class attributes:
        merged_episodes, fixed_window_size, and overlap.
        Example: Summary_Episode_20_000

        :param parameters: A dictionary with all mentioned parameters
        :return: Subfolder string variable.
        """
        # extract parameters from dict
        merged_episodes = parameters["merged_episodes"]
        fixed_window_size = parameters["fixed_window_size"]
        overlap = parameters["overlap"]

        # decide on Episodes name
        episode_name = "Summary_Episodes"
        if merged_episodes:
            episode_name = "Summary_Merged_Episodes"
        # leave 2 digits after decimal point and remove it afterward: 0.5 -> 050, 0.25 -> 025 etc.
        overlap = f"{overlap:.2f}".replace(".", "")
        return f"{episode_name}_{fixed_window_size}_{overlap}"

    @staticmethod
    def return_A_B_C_D_X_Y_path(prefix: str, parameters: dict) -> str:
        """
        Returns a path-like string. For details look into return_A_B_C_D_name and return_X_Y_name functions.
        :param prefix: Prefix of abcd folder
        :param parameters: parameters that decide about the variables.
        :return: <prefix>_A_B_C_D\\<Episode>_X_Y
        """
        abcd_folder = PathUtils.return_A_B_C_D_name(prefix, parameters)
        xy_folder = PathUtils.return_X_Y_name(parameters)
        return str(os.path.join(abcd_folder, xy_folder))

    @staticmethod
    def return_filename_from_fullpath(filepath: str, extension=False) -> str:
        """
        Returns the file name from the given file path.

        This method extracts the base name of a file from its file path. By default,
        it removes the file extension from the returned name, but the full file name
        including its extension can be returned if explicitly specified.

        :param filepath: The full path of the file from which the name should
            be extracted.
        :type filepath: str
        :param extension: A flag indicating whether the file extension should be
            included in the returned file name. Defaults to False.
        :type extension: bool, optional
        :return: The file name extracted from the file path. By default, it excludes
            the file extension. If `extension` is True, the file name with its
            extension is returned instead.
        :rtype: str
        """
        filename = os.path.basename(filepath)  # foo/bar/baz.txt -> baz.txt

        if extension:
            return filename
        else:
            return os.path.splitext(filename)[0]  # baz.txt -> baz

    @staticmethod
    def return_node_name(parameters: dict, node_type: str) -> str:
        epoch_length = parameters["fixed_window_size"]
        if node_type == "awake":
            return f"Awake_{epoch_length}"
        elif node_type == "normal_an":
            return f"Normal_ane_{epoch_length}"
        raise ValueError(f"Unknown node type {node_type}. Valid types are 'awake' and 'normal_an'")

    @staticmethod
    def filepath_exists(filepath: str) -> bool:
        return os.path.isfile(filepath)

    @staticmethod
    def list_files_in_folder(folder_path: str, extension_filter: str = None, print_to_console=False, fullpaths=False):
        """
        Lists all files in a folder, optionally filtered by file extension.

        :param folder_path: Path to the folder to list files from.
        :param extension_filter: Optional file extension to filter by (e.g. ".csv" or ".txt").
        :param print_to_console: If true will print out all files to console.
        :param fullpaths: If true returns full file paths instead of just file names.
        :returns: List of file names and the total count of these files (list of file, length of list).
        """
        if not os.path.exists(folder_path):
            print(f"Folder does not exist: {folder_path}")
            return [], 0

        all_files = [
            f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
        ]

        if extension_filter:
            all_files = [f for f in all_files if f.lower().endswith(extension_filter.lower())]

        if fullpaths:
            all_files = [os.path.join(folder_path, f) for f in all_files]

        if print_to_console:
            for file in all_files:
                print(file)

        print(f"\nTotal files{f' with extension {extension_filter}' if extension_filter else ''}: {len(all_files)}")

        return all_files, len(all_files)

    @staticmethod
    def clear_folder(folder_path: str):
        """Deletes all files in a folder of given path, not including subfolders."""

        # Nothing to do if already deleted
        if not os.path.exists(folder_path):
            print(f"Folder does not exist: {folder_path}")
            return

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):  # Deletes only files, not subdirs
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")

    @staticmethod
    def assemble_psd_file_name(start: int, end: int, result_id: int) -> str:
        """Assembles the name of a PSD file with given metadata."""
        if start is None or end is None or result_id is None:
            raise ValueError("start, end and result_id must be values")

        return f"PSD_{start}_{end}_{result_id}.csv"

    @staticmethod
    def save_file_depending_on_filetype(file_type, folder_path, file_prefix, file_suffix, result_data):
        """
        Saves a file in a specified format based on the provided file type. This function supports saving
        dataframes, dictionaries, and plots. It generates the file name using the provided prefix and
        suffix, and the file is saved in the specified folder.

        :param file_type: Type of the file to be saved. Valid options are "dataframe", "dict", and "plot".
        :param folder_path: Path of the folder where the file should be saved.
        :param file_prefix: Prefix to be used in the generated file name.
        :param file_suffix: Suffix to be used in the generated file name.
        :param result_data: Data to be saved in the file. The format of this data must align with the
            file type specified.
        :return: None
        :raises ValueError: If the provided file type is not supported.
        """
        if file_type == "dataframe":
            file_name = f"{file_prefix}_{file_suffix}.csv"
            saving_func = PathUtils.save_file_as_csv

        elif file_type == "dict":
            file_name = f"{file_prefix}_{file_suffix}.json"
            saving_func = PathUtils.save_data_as_json

        elif file_type == "plot":
            file_name = f"{file_prefix}_{file_suffix}.png"
            saving_func = PathUtils.save_plot

        else:
            raise ValueError(f"Unknown file type: {file_type}. Valid options are: dataframe, dict, plot")

        fullpath = PathUtils.return_anypath(folder_path, file_name)
        saving_func(result_data, fullpath)
        print(f"Successfully saved {file_type} to {fullpath}")

    @staticmethod
    def save_file_as_csv(data, fullpath, index=True):
        data.to_csv(fullpath, index=index)

    @staticmethod
    def save_plot(fig, fullpath: str):
        """
        Saves a matplotlib Figure to file.

        :param fig: The matplotlib Figure object to save.
        :param fullpath: Full path including filename and extension (e.g. 'figures/plot.png').
        """
        if fig is not None:
            fig.savefig(fullpath, dpi=300)
            plt.close(fig)
        else:
            print(f"Nothing to save at {fullpath}.")

    @staticmethod
    def save_data_as_json(data, fullpath):
        serial_result_data = PathUtils.serialize_for_json(data)
        with open(fullpath, "w") as f:
            json.dump(serial_result_data, f, indent=4)

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
            return {k: PathUtils.serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [PathUtils.serialize_for_json(item) for item in obj]
        else:
            return obj

    @staticmethod
    def load_json(path: str | Path) -> dict:
        """
        Loads a json file from the given path and returns its content as a dictionary.

        :param path: Path to the json file. Can be a string or a Path object.
        :return: Dictionary containing the json file's content.
        """
        path = Path(path)  # Falls ein String übergeben wurde
        with open(path, "r", encoding="utf-8") as f:
            raw_json = json.load(f)
        return PathUtils.deserialize_from_json(raw_json)

    @staticmethod
    def deserialize_from_json(obj):
        """
        Rekonstruiert Objekte, die mit serialize_for_json serialisiert wurden.
        Unterstützt pandas.DataFrame und numpy.ndarray.

        :param obj: JSON-kompatibles Objekt (dict, list, ...)
        :return: Deserialisiertes Originalobjekt (DataFrame, ndarray, ...)
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
                # Rekursiv weiter deserialisieren
                return {k: PathUtils.deserialize_from_json(v) for k, v in obj.items()}

        elif isinstance(obj, list):
            return [PathUtils.deserialize_from_json(item) for item in obj]

        else:
            return obj

    import pandas as pd
    from pathlib import Path

    @staticmethod
    def append_unique_rows_to_csv(df: pd.DataFrame, csv_path: str | Path) -> tuple[pd.DataFrame, int]:
        """
        Appends a DataFrame to a CSV file, ensuring no duplicate rows are written.
        If the CSV does not exist, it will be created.

        Assumes that all rows in the DataFrame and CSV have the same structure (i.e., same columns).

        :param df: DataFrame to append.
        :param csv_path: Path to the target CSV file.
        :return: A tuple containing the DataFrame with the new rows and the number of new rows added.
        """
        csv_path = Path(csv_path)

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

    @staticmethod
    def create_dir_path(path):
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
        else:
            print(f"Directory already exists: {path}")
