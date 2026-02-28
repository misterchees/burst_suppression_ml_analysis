"""Module for the LoadData class, and a psd load function"""
import json
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd

from MachineLearning.IO.io_core import IOCore
from MachineLearning.Utils.file_data_utils import FileDataUtils
from MachineLearning.Utils.path_utils import PathUtils


def load_psd_with_start_end_resultid(folder_path: Path, filename: str) \
        -> Tuple[pd.DataFrame, int, int, int]:
    """
    Loads a PSD csv file from the given folder as Dataframe and returns it with metadata extracted from the filename.
    :param filename: A file with this name structure -> start_end_resultid.csv
    :param folder_path: The path to the folder of filename
    :return: A tuple structured in the following way (dataframe, start, end, result_id)
    """
    psd_fullpath = Path(folder_path, filename)
    print(f"Processing {psd_fullpath}")

    # Validation of single episode PSD name structure
    name_parts = filename.split(".")[0].split("_")
    if len(name_parts) != 4 and name_parts[0] != "PSD":
        raise ValueError(f"Name of file {filename} has no typical structure for single episode PSD."
                         "Typical structure example for reference: PSD_0_1_2.csv")

    # Get each name part and cast to int
    metadata = filename.replace(".csv", "").split("_")[1:]  # PSD_0_1_2.csv -> ['0','1','2']
    start = int(float(metadata[0]))
    end = int(float(metadata[1]))
    result_id = int(metadata[2])
    psd_dataframe = pd.read_csv(psd_fullpath)

    return psd_dataframe, start, end, result_id


class LoadData(IOCore):
    """This class handles the loading of the data in the project"""

    def __init__(self):
        """Initializes the class with configs from IOCore superclass."""
        super().__init__()

    def load_faw_times_as_df(self, parameters: dict) -> pd.DataFrame:
        """
        Assembles a path to the csv-file of with the times of the windows depending on passed parameters
        in the Fake-Awake (FAW) directory and loads it into a Pandas DataFrame.
        :param parameters: A dictionary with all episode parameters from the project
        :return: A pandas DataFrame containing the episodes based on the parameters passed.
        """
        # Assemble the fullpath to the CSV file
        faw_dir = self.return_path_info(["faw"])
        a_b_c_d_dir = PathUtils.return_A_B_C_D_path("result", parameters)
        x_y_name = PathUtils.return_X_Y_name(parameters)
        csv_fullpath = Path(faw_dir, a_b_c_d_dir, f"{x_y_name}.csv")
        # Validate fullpath
        if not csv_fullpath.is_file():
            raise FileNotFoundError(f"CSV not found: {csv_fullpath}")
        # Read CSV to DataFrame
        df = pd.read_csv(csv_fullpath)
        return df

    def load_awake_times_as_df(self, parameters: dict, awake_cleaned: bool = True, transition_time=10) -> pd.DataFrame:
        """
        Reads a CSV file with 'caseid' and 'anestart' columns and generates epochs
        based on a fixed epoch length.

        :param parameters: Parameters for epochs. Contains the length of each epoch.
        :param awake_cleaned: A boolean to indicate if the episodes should be taken from the awake_cleaned.txt
        :param transition_time: The transition time for the patient to respond to anesthesia beginning.
        :returns: A DataFrame with columns ['Start', 'End', 'ResultID'] representing the epochs.
        """
        if awake_cleaned:
            return self.load_cleaned_awake_times_as_df(parameters)

        csv_path = self.return_csv_path_from_basedir("awake_times")
        input_df = pd.read_csv(csv_path)
        epoch_length = int(parameters["fixed_window_size"])

        all_epochs = []

        for _, row in input_df.iterrows():
            caseid = row['caseid']
            anestart = int(row['anestart'])
            num_epochs = int(
                (anestart - transition_time) // epoch_length)  # segment into epochs based on episode length

            for i in range(num_epochs):
                start = i * epoch_length
                end = start + epoch_length
                all_epochs.append({
                    'Start': start,
                    'End': end,
                    'ResultID': caseid
                })

        return pd.DataFrame(all_epochs)

    @staticmethod
    def load_cleaned_awake_times_as_df(parameters: dict) -> pd.DataFrame:
        """
        Basically does the same as load_awake_times_as_df. The difference is the start of AW epochs is not zero
        and the input file path is hardcoded.
        :param parameters: Parameters for epochs. Contains the length of each epoch.
        :return: A DataFrame with columns ['Start', 'End', 'ResultID'] representing the epochs.
        """
        csv_path = r"E:\Daten\awake_cleaned.txt"
        input_df = pd.read_csv(csv_path)
        epoch_length = int(parameters["fixed_window_size"])

        all_epochs = []

        for _, row in input_df.iterrows():
            caseid = int(row['case_id'])
            epoch_start = float(row['start_time'])
            epoch_end = float(row['end_time'])
            num_epochs = int(
                (epoch_end - epoch_start) // epoch_length)  # segment into epochs based on episode length

            for i in range(num_epochs):
                start = epoch_start + (i * epoch_length)
                end = start + epoch_length
                all_epochs.append({
                    'Start': start,
                    'End': end,
                    'ResultID': caseid
                })

        return pd.DataFrame(all_epochs)


    def sample_anesthesia_epochs(self, parameters: dict, num_epochs: int, transition_sec: int = 10,
                                 safety_margin_min: int = 10, random_state: int = 42,
                                 epochs_per_eeg: int = 30) -> pd.DataFrame:
        """
        Samples random EEG epochs from anesthesia segments (i.e., neither awake nor FAW).

        :param parameters: Defines length of each epoch.
        :param num_epochs: Number of total epochs to sample.
        :param transition_sec: Time (in seconds) after anesthesia starts before epochs are allowed.
        :param safety_margin_min: Minutes to exclude from the end of EEG to avoid flatline segments.
        :param random_state: Random seed for reproducibility.
        :param epochs_per_eeg: Max number of epochs to sample per eeg
                                (Fewer samples per patient -> better distribution).
        :returns: DataFrame with columns Start, End, ResultID
        """
        import random

        # set initial data
        random.seed(random_state)
        anestart_csv_path = self.return_csv_path_from_basedir("awake_times")
        epoch_length_sec = int(parameters["fixed_window_size"])
        filtered_data_dir = self.return_path_info(["filtered_data"])

        anestart_df = pd.read_csv(anestart_csv_path)
        result = []

        # List of all available EEG files
        eeg_files = [f for f in filtered_data_dir.iterdir() if f.suffix =='.csv']
        random.shuffle(eeg_files)

        # Look into EEG files and sample anesthesia episodes with the specified parameters
        for eeg_file in eeg_files:
            result_id = eeg_file.stem  # 1.csv -> ['1', '.csv'] -> 1
            eeg_path = Path(filtered_data_dir, eeg_file)
            # Look up anestart for patient ID
            match = anestart_df.loc[anestart_df['caseid'].astype(str) == result_id, 'anestart']
            if not match.empty:
                anestart = match.values[0]
            else:
                # Default to safety margin if anestart is not available for this ID
                anestart = safety_margin_min * 60

            # Get duration of the EEG file
            with open(eeg_path) as f:
                num_lines = sum(1 for _ in f) - 1  # minus header
            eeg_duration = num_lines // 128  # assuming 128 Hz sampling rate

            # Compute valid range for sampling
            start_limit = anestart + transition_sec
            end_limit = eeg_duration - (safety_margin_min * 60) - epoch_length_sec
            if end_limit <= start_limit:
                continue  # skip files where there's not enough space

            # Skip files that are too short to retrieve an epoch
            max_possible_epochs = (end_limit - start_limit) // epoch_length_sec
            if max_possible_epochs <= 0:
                continue

            # Sample max is constrained by length of EEG, number of the necessary epochs, or given number per EEG
            # The smallest of these three will be picked each iteration
            num_to_sample = min(max_possible_epochs, num_epochs - len(result), epochs_per_eeg)
            start_points = random.sample(range(start_limit, end_limit - epoch_length_sec + 1), num_to_sample)

            for start in start_points:
                result.append({
                    "Start": start,
                    "End": start + epoch_length_sec,
                    "ResultID": result_id
                })

            if len(result) >= num_epochs:
                break  # Exit if we've collected enough

        return pd.DataFrame(result)

    def load_grouped_epochs(self, parameters: dict, epoch_type: str, num_epochs: int = None
                            ) -> Dict[int, List[Tuple[int, int]]]:
        """
        Returns EEG snippets (Epochs) based on given parameters.
        :param parameters: Parameters for episodes containing primary parameters for chosen faw eopchs.
        :param epoch_type: Defines epoch times.
        :param num_epochs: Number of anesthesia epochs. Will be ignored if epoch_type is not 'normal_an'
        :return: Dictionary {ResultID: [(start1, end1), (start2, end2), ...]}
        """
        # Retrieve epochs based on their epoch type
        if epoch_type == 'awake':
            epoch_times_df = self.load_awake_times_as_df(parameters)
        elif epoch_type == 'faw':
            epoch_times_df = self.load_faw_times_as_df(parameters)
        elif epoch_type == 'normal_an':
            epoch_times_df = self.sample_anesthesia_epochs(parameters, num_epochs=num_epochs)
        else:
            raise ValueError(f'Epoch type "{epoch_type}" not recognized. Valid options: "awake", "faw", "normal_an"')

        # Group and return epochs
        grouped_epoch_times = self.group_epochs_by_result_id(epoch_times_df)
        return grouped_epoch_times

    def load_eeg_data(self, result_id: int, filtered=True) -> Tuple[int, np.ndarray]:
        """
        Assembles a path to the EEG File of interest, specified by the patient ID.
        Returns fs and raw EEG as a Tuple.

        :param result_id: The patient ID
        :param filtered: If True retrieves the filtered EEG file instead of the raw EEG file
        :return: a tuple (fs, eeg). fs -> sampling frequency; eeg -> an EEG samples array with two channels
        """
        if filtered:
            fs, eeg = self._load_filtered_eeg_data(result_id)
        else:
            fs, eeg = self._load_raw_eeg_data(result_id)
        return fs, eeg

    def _load_raw_eeg_data(self, result_id: int) -> Tuple[int, np.ndarray]:
        """
        Assembles a path to the raw EEG .mat file of interest, specified by the patient ID and
        returns fs and raw EEG as a Tuple

        :param result_id: The patient ID
        :return: a tuple (fs, eeg). fs -> sampling frequency; eeg -> a raw-EEG samples array with two channels
        """
        # Assemble Path to directory with .mat files
        vitaldb_eeg_dir = self.return_path_info(["initial_data", "raw_eeg_mat"])

        mat_file_path = Path(vitaldb_eeg_dir, f"{result_id}.mat")
        if not mat_file_path.is_file():
            return self._load_eeg_vitaldb_csv(result_id)

        # Scipy import to read .mat files
        import scipy.io

        # load .mat file
        eeg_cols = self.data_names["eeg_files"]
        mat_data = scipy.io.loadmat(str(mat_file_path))
        fs = int(mat_data[eeg_cols["eeg_fs"]].squeeze())
        raw_eeg = mat_data[eeg_cols["eeg_rawEEG"]]

        return fs, raw_eeg

    def _load_eeg_vitaldb_csv(self, result_id: int) -> Tuple[int, np.ndarray]:
        """
        Loads EEG data from a CSV file and performs basic pre-processing operations such as handling missing values.
        The EEG data is extracted from specified columns, interpolated to handle NaN values, and returned as a clean
        NumPy array along with the sampling frequency.

        :param result_id: Identifier of the specific EEG result whose data needs to be loaded. The file name is
                          derived from this ID in the format `{result_id}.csv`.

        :return: A tuple containing the sampling frequency as an integer and the pre-processed EEG
                data as a NumPy array.

        :raise FileNotFoundError: If the specified CSV file corresponding to the `result_id` is not found
                                in the expected directory.
        """
        # Assemble Path to directory with .mat files
        vitaldb_eeg_dir = self.return_path_info(["initial_data", "raw_eeg_mat"])
        csv_file_path = Path(vitaldb_eeg_dir, f"{result_id}.csv")
        if not csv_file_path.is_file():
            raise FileNotFoundError(f"File found: {csv_file_path}")

        fs = 128
        csv_file = pd.read_csv(csv_file_path)
        raw_eeg = csv_file[['BIS/EEG1_WAV', 'BIS/EEG2_WAV']].to_numpy()
        # --- Pre-processing: Handle NaN values ---
        # Convert to DataFrame to use pandas' powerful interpolation methods
        # This fixes the issue where NaNs cause the filter to output only NaNs
        df_raw = pd.DataFrame(raw_eeg)

        # Linear interpolation fills gaps based on surrounding values
        # limit_direction='both' ensures NaNs at the very beginning or end are also filled
        df_interpolated = df_raw.interpolate(method='linear', axis=0, limit_direction='both')

        # Convert back to ndarray
        clean_eeg = df_interpolated.to_numpy()

        return fs, clean_eeg


    def _load_filtered_eeg_data(self, result_id: int) -> Tuple[int, np.ndarray]:
        """
        Loads a filtered EEG from a CSV file with a comment line containing the sampling rate (fs).
        :param result_id: The patient ID
        :returns: a tuple (fs, eeg). fs -> sampling frequency; eeg -> a filtered-EEG samples array with two channels
        """
        # Assemble Path to directory with filtered-EEG files
        filtered_eeg_dir = self.return_path_info(["filtered_data"])

        filepath = Path(filtered_eeg_dir, f"{result_id}.csv")
        if not filepath.is_file():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
            if not first_line.startswith("# fs = "):
                raise ValueError("First line does not contain sampling rate in expected format: '# fs = <int>'")
            fs = int(first_line.replace("# fs = ", "").strip())

        # Load EEG data, skipping the first line (comment line)
        df = pd.read_csv(filepath, skiprows=1)
        eeg_cols = self.data_names["eeg_files"]
        raw_eeg = df[eeg_cols["eeg_channels"]].to_numpy()

        return fs, raw_eeg

    def load_eeg_epochs_from_csv(self, result_id: int, epochs: list, channel: int, folder_keys: List[str])\
            -> Tuple[int, Dict[Tuple[int, int], np.ndarray]]:
        """
        Reads EEG segments selected by the information in epochs for a given channel from a CSV file.
        Assuming the file has a header comment containing the sampling rate.

        :param result_id: Patient ID to determine the path to the EEG CSV file.
        :param epochs: List of (start_time, end_time) tuples in seconds.
        :param channel: EEG channel to extract (1 or 2).
        :param folder_keys: List of folder keys to determine the path to the EEG CSV file.
        :returns: Tuple of (sampling rate as int, dict of (start, end): EEG segment as ndarray)
        """
        # Assemble Path to directory with .csv files
        eeg_dir = self.return_path_info(folder_keys)

        filepath = Path(eeg_dir, f"{result_id}.csv")
        if not filepath.is_file():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Read sampling rate (fs) from the first comment line
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
            if not first_line.startswith("# fs = "):
                raise ValueError("Missing or malformed sampling rate comment line.")
            fs = int(first_line.replace("# fs = ", "").strip())

        # Compute required data row indices (account for header lines)
        index_offset = 2  # One comment line and one header line
        required_rows = set()
        sample_ranges = []

        for start, end in epochs:
            start_row = int(start * fs) + index_offset
            end_row = int(end * fs) + index_offset
            sample_ranges.append((start_row, end_row))
            required_rows.update(range(start_row, end_row))

        # Build row-skipping function for pandas (for efficiency)
        def skiprows(i):
            return i != 1 and i not in required_rows  # only keeps header line and required rows

        # Read only selected rows and the requested channel
        col = str(channel)  # 1 -> '1'
        df = pd.read_csv(filepath, usecols=[col], skiprows=skiprows)

        # Extract segments from the DataFrame
        values = df[col].to_numpy()
        segments = {}
        cursor = 0

        # create output dict as segments. Looks like this: {(start1, end1): segment1, ..., (startN, endN): segmentN}
        for (start_sec, end_sec), (start_row, end_row) in zip(epochs, sample_ranges):
            num_samples = end_row - start_row
            segment = values[cursor:cursor + num_samples]
            segments[(start_sec, end_sec)] = segment
            cursor += num_samples

        return fs, segments

    @staticmethod
    def group_epochs_by_result_id(all_epochs_df: pd.DataFrame) -> Dict[int, List[Tuple[int, int]]]:
        """
        Groups epoch start/end times by ResultID into a dictionary.

        :param all_epochs_df: DataFrame with columns 'Start', 'End', 'ResultID'
        containing epochs from different patients
        :returns: Dictionary {ResultID: [(start1, end1), (start2, end2), ...]}
        """
        grouped = (
            all_epochs_df.groupby("ResultID")[["Start", "End"]]
            .apply(lambda x: [tuple(row) for row in x.to_numpy()])
            .to_dict()
        )
        return grouped

    def return_csv_path_from_basedir(self, file_key: str) -> Path:
        """Takes the path_config key of a file in the base directory and returns a path to that file."""
        base_dir = self.path_config["base_dir"]["path_name"]
        file_name = self.path_config["base_dir"]["files"][file_key]
        file_path = Path(base_dir, f"{file_name}.csv")
        return file_path

    def load_model(self, model_key: str, parameters: dict):
        """
        Loads a model from a specified path, defined by model_key and parameters.
        :param model_key: Key of the model (folder)
        :param parameters: Parameters for the model -> define the subfolder names.
        """

        from joblib import load
        full_folder_path = self.return_all_parameter_fullpath(parameters, False, True, ["models", model_key])
        model_file = f"{model_key}.joblib"
        model_fullpath = Path(full_folder_path, model_file)
        model = load(model_fullpath)
        return model

    def load_metrics(self, parameters: dict, model_key: str, run_name: str = None ) -> dict:
        """
        Loads the metrics from a JSON file located in a specified directory, which path is
        computed dynamically based on provided parameters, model key, and an optional run name.


        :param parameters: Dictionary containing configuration details required to construct the file path.
        :param model_key: The specific model identifier, used to locate corresponding result directories.
        :param run_name: The specific run name to further narrow down the folder path, if provided.

        :return: A dictionary containing the loaded metrics from the JSON file.
        """
        # Construct the path to the folder from where the file will be loaded
        folder_path = self.return_all_parameter_fullpath(
            parameters, False, True, ["results", model_key], run_name)
        file_name = f"folds_metrics.json"
        fullpath = Path(folder_path, file_name)

        return self.load_json(fullpath)

    def load_metadata_file(self, parameters: dict, model_key: str, filename: str, outlier_run_name: str | None):
        """
        Loads a metadata file based on the given parameters, model key, and filename.

        :param parameters: A dictionary containing the hyperparameters used to
                            construct the file path.
        :param model_key: The key associated with a specific model, used as an identifier while
                            forming the directory path for the metadata file.
        :param filename: The name of the metadata file to be loaded, including its extension.
        :param outlier_run_name: The name of the outlier run to load metadata for.
                                If None, loads metadata for the current run_name.
        :return: Parsed data from the metadata file. The return type depends on the file
                format: a JSON object for `.json` files or a pandas DataFrame for `.csv` files.
        :raises ValueError: If the files extension is not supported (i.e., not "json" or "csv").
        """
        # Construct the path to the folder from where the file will be loaded
        folder_path = self.return_all_parameter_fullpath(
            parameters, False, False, ["metadata_analysis", model_key], outlier_run_name
        )
        fullpath = Path(folder_path, filename)

        # Load file based on extension
        extension = filename.split(".")[-1]
        if extension == "json":
            return self.load_json(fullpath)
        elif extension == "csv":
            return pd.read_csv(fullpath)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")


    def load_problematic_ids(self, parameters: dict, model_key: str, outlier_run_name: str | None,
                             global_outliers: bool) -> list | None:
        """
        Loads the IDs of outlier groups related to a specific model, from a metadata file.
        If the specified file does not exist, attempts to create it from previous analysis results.

        :param parameters: A dictionary containing parameter configurations for the model.
        :param model_key: The key or identifier of the model for which problematic IDs are being loaded.
        :param outlier_run_name: The name of the outlier run to load IDs for.
                                    If None, loads IDs for the current run_name.
        :param global_outliers: Flag indicating whether to include outliers from all runs.
        :return: A list of outlier groups extracted from the metadata file.

        :raises FileNotFoundError: If the metadata file does not exist and cannot be created from
                                    previous analysis results.
        """

        print("Loading IDs of outlier groups...")
        try:
            outliers_df = self.load_metadata_file(
                parameters, model_key, "Summary_outliers_by_groups.csv", outlier_run_name
            )
        except FileNotFoundError:
            print("File 'Summary_outliers_by_groups.csv' not found. Trying to create from previous results...")
            try:
                from MachineLearning.Evaluation.meta_fold_analyzer import MetaFoldAnalyzer
                fold_analyzer = MetaFoldAnalyzer(model_key, parameters, outlier_run_name)
                outliers_df = fold_analyzer.select_outlier_groups(save_res=True)
            except FileNotFoundError:
                print("No previous results found to create 'Summary_outliers_by_groups.csv'. "
                      "No problematic IDs can be loaded.")
                return None

        # Ensure that global outliers include outliers from the given run by saving them before loading global outliers
        if global_outliers:
            outliers_df = self._update_and_get_global_outliers(parameters, outliers_df, "patient_id")

        outlier_list = outliers_df["group"].values.tolist()
        print(f"Outlier groups that will be removed from analysis: {outlier_list}")

        return outlier_list

    def load_problematic_epochs(self, parameters: dict, model_key: str, outlier_run_name: str | None,
                                global_outliers: bool) -> pd.DataFrame | None:
        """
        Loads problematic epochs based on metadata or by analyzing previous results. If global outliers are
        requested, they will include the identified outliers from the current run before global outliers are loaded.
        The function returns the outlier epochs in the form of a DataFrame or None if no problematic epochs are found.

        :param parameters: Configuration parameters used in the operation.
        :param model_key: Unique identifier for the model being used.
        :param outlier_run_name: Optional identifier for the run to load outlier metadata for.
        :param global_outliers: Flag indicating whether to include global outliers in the analysis.
        :return: A DataFrame containing the problematic outlier epochs or None if no problematic epochs are found.
        """

        print("Loading outlier epochs...")
        try:
            outliers_df = self.load_metadata_file(
                parameters, model_key, "Summary_outlier_epochs.csv", outlier_run_name
            )
        except FileNotFoundError:
            print("File 'Summary_outlier_epochs.csv' not found. Trying to create from previous results...")
            try:
                from MachineLearning.Evaluation.meta_fold_analyzer import MetaFoldAnalyzer
                fold_analyzer = MetaFoldAnalyzer(model_key, parameters, outlier_run_name)
                outliers_df = fold_analyzer.select_outlier_epochs(save_res=True)
            except FileNotFoundError:
                print("No previous results found to create 'Summary_outlier_epochs.csv'. "
                      "No problematic epochs can be loaded.")
                return None

        # Ensure that global outliers include outliers from given run by saving them before loading global outliers
        if global_outliers:
            outliers_df = self._update_and_get_global_outliers(parameters, outliers_df, "epoch")

        outlier_list = outliers_df.values.tolist()
        print(f"Outlier epochs that will be removed from analysis: {outlier_list}")

        return outliers_df

    def load_run_data(self, hyperparameters: dict, run_name: str, model_key: str) -> dict:
        """Loads the run data from a run specified by name, hyperparameters, and model_key."""
        metadata_path = self.return_run_metadata_fullpath(hyperparameters, run_name, model_key)
        metadata = self.load_json(metadata_path)
        return metadata

    def load_splits(self, hyperparamers: dict, run_name: str, combined=True) -> dict|pd.DataFrame:
        """
        Loads dataset splits from specified file paths into a dictionary of DataFrames or a combined DataFrame.

        :param hyperparamers: Configuration dictionary specifying parameters
                              required to determine dataset split file paths.
        :param run_name: Name of the run used to access associated split file paths.
        :param combined: Flag to determine whether all splits should be combined
                         into a single DataFrame. Defaults to True.
        :returns: A single combined DataFrame if 'combined' is True. Otherwise, a dictionary
                  of DataFrames indexed by the file name (stem) of each split.
        """
        splits_list = self.return_related_fullpaths(hyperparamers, run_name, ["test_and_train_data", "splits"])

        df_dict = {}
        for split_path in splits_list:
            df_dict[split_path.stem] = pd.read_csv(split_path)

        if combined:
            df_combined = pd.concat(df_dict.values(), axis=0)
            return df_combined
        else:
            return df_dict

    def load_results(self, hyperparamers: dict, run_name: str, model_key: str, combined=True) -> dict|pd.DataFrame:
        """
        Loads experiment results based on specified hyperparameters, run name, and model key
        into a dictionary of DataFrames or a combined DataFrame

        :param hyperparamers: Dictionary containing hyperparameter values used to filter
                            the related results.
        :param run_name: Name of the experiment run to identify the result files.
        :param model_key: Specific model key to filter and locate result files.
        :param combined: Flag indicating whether to return a single concatenated
                        DataFrame (True) or a dictionary of individual DataFrames
                        (False). Defaults to True.
        :return: If combined is True, returns a pandas DataFrame combining all individual result DataFrames.
                If combined is False, returns a dictionary where keys are file names and values
                are the corresponding result DataFrames.
        """
        results_list = self.return_related_fullpaths(hyperparamers, run_name, ["results", model_key])

        df_dict = {}
        for result_path in results_list:
            df_dict[result_path.stem] = pd.read_csv(result_path)

        if combined:
            df_combined = pd.concat(df_dict.values(), axis=0)
            return df_combined
        else:
            return df_dict

    def load_combined_features_df(self, parameters: dict, class_1: str, class_0: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load combined features dataframes for the specified classes.

        This method loads data from specified file paths for two given classes and
        returns their corresponding dataframes. It uses the provided parameters to
        determine file paths for the data of `class_1` and `class_0`. The paths are
        generated dynamically based on the parameters and predefined folder hierarchy.

        :param parameters: A dict containing configuration parameters.
        :param class_1: The name or identifier of the first class whose data will be loaded.
            Valid options are: 'normal_an', 'faw' and 'awake'
        :param class_0: The name or identifier of the second class whose data will be loaded.
            Valid options are: 'normal_an', 'faw' and 'awake'
        :return: A tuple of pandas DataFrames corresponding to the data of `class_1` and `class_0`.
        """
        class_1_path = self.return_file_fullpath(parameters, True, False, class_1,
                                                              ["test_and_train_data", "feature_sets"])
        class_0_path = self.return_file_fullpath(parameters, True, False, class_0,
                                                              ["test_and_train_data", "feature_sets"])
        print(f"Loading {class_1} data from {class_1_path}\n "
              f"Loading {class_0} data from {class_0_path}")
        class_1_df = pd.read_csv(class_1_path).copy() # Copying to avoid pd messing with the file content
        class_0_df = pd.read_csv(class_0_path).copy()

        return class_1_df, class_0_df

    def load_global_outliers(self, parameters, outlier_type) -> pd.DataFrame:
        """
        Loads a dataframe of global outliers based on the specified outlier type and parameters.
        The method determines the target file name based on the `outlier_type` and retrieves
        the file location using the provided parameters. It reads and returns the data from
        the corresponding CSV file.

        :param parameters: List of parameters used to determine the folder path for outliers files.
        :param outlier_type: Specifies the type of outliers to fetch, either "epoch" or "patient_id".
        :return: Pandas DataFrame containing the data of global outliers.
        :raises ValueError: If the provided outlier_type is not "epoch" or "patient_id".
        """
        folder_path = self.return_all_parameter_fullpath(parameters, False, False, ["global_outliers"])

        # Build a fullpath depending on the outlier type
        if outlier_type == "epoch":
            filename = "global_epoch_outliers.csv"
        elif outlier_type == "patient_id":
            filename = "global_patient_outliers.csv"
        else:
            raise ValueError("Invalid outlier type. Expected 'epoch' or 'patient_id'.")

        fullpath = Path(folder_path, filename)
        # Load and return outlier df
        outliers_df = pd.read_csv(fullpath)
        return outliers_df

    def load_further_results(self, hyperparameters: dict, analysis_key: str, result_type: str, filename: str)\
            -> pd.DataFrame|dict:
        """
        Loads additional results specified by the analysis key, result type, and filename from
        the corresponding path derived using the given hyperparameters.

        This method determines the full path to the results based on the hyperparameters
        and loads the data in either a DataFrame or JSON structure, as specified by the result
        type. It raises an exception if the result type is invalid.


        :param hyperparameters: Dictionary containing the hyperparameters which are used
            to generate the full path to the results.
        :param analysis_key: The Key, used to identify the specific analysis folder within the
            hyperparameters structure.
        :param result_type: The type of results to load. Must be either 'dataframe' (loads
            using pandas) or 'json' (loads using the custom JSON loader).
        :param filename: The name of the file to be loaded from the derived folder path.

        :returns: The loaded results as a pandas DataFrame if the type is
            'dataframe', or as a dictionary if the type is 'json'.

        :raises ValueError: If the `result_type` is not 'dataframe' or 'json'.
        """
        folder_path = self.return_all_parameter_fullpath(hyperparameters, False, True,
                                                         ["further_analysis", analysis_key])

        if result_type == "dataframe":
            Path(folder_path, filename)
            return pd.read_csv(Path(folder_path, filename))
        elif result_type == "json":
            return self.load_json(Path(folder_path, filename))
        else:
            raise ValueError(f"Invalid result type. Expected 'dataframe' or 'json', got {result_type}")

    def _update_and_get_global_outliers(self, parameters, outliers_df: pd.DataFrame, outlier_type: str) -> pd.DataFrame:
        """
        Updates the global outliers with new outliers and retrieves the updated global outliers.
        This function saves the new outliers to the global outliers using an external saver instance
        and then fetches the updated global outliers dataset.

        :param parameters: The parameters required for saving and loading global outliers.
        :param outliers_df: A pandas DataFrame containing the new outliers to be added to the global outliers.
        :param outlier_type: A string representing the specific type of outlier
            valid options are: "patient_id", "epoch".
        :return: A pandas DataFrame containing the updated global outliers after the addition of the new outliers.
        """
        # Create the saver instance
        from MachineLearning.IO.save_result import SaveResult
        saver = SaveResult()

        saver.save_global_outliers(parameters, outliers_df, outlier_type)  # Add given outliers to global
        global_outliers_df = self.load_global_outliers(parameters, outlier_type)  # Get updated global outliers
        return global_outliers_df
    
    @staticmethod
    def load_json(path: Path) -> dict:
        """
        Loads a JSON file from the given path and returns its content as a dictionary.

        :param path: Path to the JSON file.
        :return: Dictionary containing the JSON file's content.
        """
        with open(path, "r", encoding="utf-8") as f:
            raw_json = json.load(f)
        return FileDataUtils.deserialize_from_json(raw_json)