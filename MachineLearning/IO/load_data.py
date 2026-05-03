"""Module for the LoadData class, and a psd load function"""
import json
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd

from MachineLearning.Utils.config_handler import load_config
from MachineLearning.Utils.path_manager import PathManager
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
    psd_fullpath = folder_path / filename
    print(f"Processing {psd_fullpath}")

    # Validation of single episode PSD name structure
    name_parts = filename.split(".")[0].split("_")
    if len(name_parts) != 4 or name_parts[0] != "PSD":
        raise ValueError(f"Name of file {filename} has no typical structure for single episode PSD."
                         "Typical structure example for reference: PSD_0_1_2.csv")

    # Get each name part and cast to int
    metadata = filename.replace(".csv", "").split("_")[1:]  # PSD_0_1_2.csv -> ['0','1','2']
    start = int(float(metadata[0]))
    end = int(float(metadata[1]))
    result_id = int(metadata[2])
    psd_dataframe = pd.read_csv(psd_fullpath)

    return psd_dataframe, start, end, result_id


class LoadData:
    """This class handles the loading of the data in the project"""

    def __init__(self, pm: PathManager):
        """Initializes the class."""
        self.data_names = load_config("data_names_config.yaml")
        self.pm = pm

    def load_faw_times_as_df(self, parameters: dict) -> pd.DataFrame:
        """
        Assembles a path to the csv-file of with the times of the windows depending on passed parameters
        in the Fake-Awake (FAW) directory and loads it into a Pandas DataFrame.
        :param parameters: A dictionary with all episode parameters from the project
        :return: A pandas DataFrame containing the episodes based on the parameters passed.
        """
        # Assemble the fullpath to the CSV file
        faw_dir = self.pm.get_path("faw")
        a_b_c_d_dir = PathUtils.return_A_B_C_D_path("result", parameters)
        x_y_name = PathUtils.return_X_Y_name(parameters)
        csv_fullpath = faw_dir / a_b_c_d_dir / f"{x_y_name}.csv"
        # Validate fullpath
        if not csv_fullpath.is_file():
            raise FileNotFoundError(f"CSV not found: {csv_fullpath}")
        # Read CSV to DataFrame
        df = pd.read_csv(csv_fullpath)
        return df

    def load_awake_times_as_df(self, parameters: dict, awake_cleaned: bool = False, transition_time=10) -> pd.DataFrame:
        """
        Reads a CSV file with 'caseid' and 'anestart' columns and generates epochs
        based on a fixed epoch length.

        :param parameters: Parameters for epochs. Contains the length of each epoch.
        :param awake_cleaned: A boolean to indicate if the episodes should be taken from the awake_cleaned.txt
        :param transition_time: The transition time for the patient to respond to anesthesia beginning.
        :returns: A DataFrame with columns ['Start', 'End', 'ResultID'] representing the epochs.
        """
        file_key = "cleaned_aw_times" if awake_cleaned else "awake_times"

        csv_path = self.pm.get_path(file_key)
        epoch_length = int(parameters["fixed_window_size"])
        input_df = pd.read_csv(csv_path)

        all_epochs = []

        for _, row in input_df.iterrows():
            caseid = int(row['case_id']) if awake_cleaned else row['caseid']

            # Calculate number of epochs accordingly to file information (normal ane always starts at 0)
            epoch_end = float(row['end_time']) if awake_cleaned else int(row['anestart'])
            epoch_start = float(row['start_time']) if awake_cleaned else 0
            transition_time = 0 if awake_cleaned else transition_time
            num_epochs = int(
                (epoch_end - epoch_start - transition_time) // epoch_length)

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
        anestart_csv_path = self.pm.get_path("awake_times")
        epoch_length_sec = int(parameters["fixed_window_size"])
        filtered_data_dir = self.pm.get_path("filtered_data")

        anestart_df = pd.read_csv(anestart_csv_path)
        result = []

        # List of all available EEG files
        eeg_files = [f for f in filtered_data_dir.iterdir() if f.suffix =='.csv']
        random.shuffle(eeg_files)

        # Look into EEG files and sample anesthesia episodes with the specified parameters
        for eeg_file in eeg_files:
            result_id = eeg_file.stem  # 1.csv -> ['1', '.csv'] -> 1
            eeg_path = filtered_data_dir / eeg_file
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
            eeg_duration = num_lines // 128  # Time in seconds assuming 128 Hz sampling rate

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
        grouped_epoch_times = self._group_epochs_by_result_id(epoch_times_df)
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
        vitaldb_eeg_dir = self.pm.get_path("initial_data", "raw_eeg_mat")

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

    def _load_filtered_eeg_data(self, result_id: int) -> Tuple[int, np.ndarray]:
        """
        Loads a filtered EEG from a CSV file with a comment line containing the sampling rate (fs).
        :param result_id: The patient ID
        :returns: a tuple (fs, eeg). fs -> sampling frequency; eeg -> a filtered-EEG samples array with two channels
        """
        # Assemble Path to directory with filtered-EEG files
        filtered_eeg_dir = self.pm.get_path("filtered_data")

        filepath = filtered_eeg_dir / f"{result_id}.csv"
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

    def _load_eeg_vitaldb_csv(self, result_id: int) -> Tuple[int, np.ndarray]:
        """
        Loads EEG data from a CSV file and performs basic pre-processing operations.
        The EEG data is extracted from specified columns, interpolated to handle NaN values, and returned as a
        NumPy array along with the sampling frequency (Which is always 128 Hz for VitalDB EEG data).

        :param result_id: Identifier of the EEG track. It is derived from the file name `{result_id}.csv`.

        :return: A tuple containing the sampling frequency as an integer and the pre-processed EEG
                data as a NumPy array.

        :raise FileNotFoundError: If the specified CSV file corresponding to the `result_id` is not found
                                in the expected directory.
        """
        # Assemble Path to directory with .mat files
        vitaldb_eeg_dir = self.pm.get_path("initial_data", "raw_eeg_mat")
        csv_file_path = vitaldb_eeg_dir / f"{result_id}.csv"
        if not csv_file_path.is_file():
            raise FileNotFoundError(f"File not found: {csv_file_path}")

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
        eeg_dir = self.pm.get_path(*folder_keys)

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

        # Build a row-skipping function for pandas (for efficiency)
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
    def _group_epochs_by_result_id(all_epochs_df: pd.DataFrame) -> Dict[int, List[Tuple[int, int]]]:
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

    def load_model(self, model_key: str, parameters: dict):
        """
        Loads a model from a specified path, defined by model_key and parameters.
        :param model_key: Key of the model (folder)
        :param parameters: Parameters for the model -> define the subfolder names.

        :return: The loaded model
        """

        from joblib import load
        full_folder_path = self.pm.get_complex_ml_path(
            parameters, ["models", model_key], False, True
        )
        model_file = f"{model_key}.joblib"
        model_fullpath = full_folder_path / model_file
        model = load(model_fullpath)
        return model

    def load_outliers(self, parameters: dict, model_key: str, outlier_run_name: str | None,
                             global_outliers: bool, grouped_by_id: bool) -> list | None:
        """
                Loads the IDs of outlier groups related to a specific model, from a metadata file.
                If the specified file does not exist, attempts to create it from previous analysis results.

                :param parameters: A dictionary containing parameter configurations for the model.
                :param model_key: The key or identifier of the model for which problematic IDs are being loaded.
                :param outlier_run_name: The name of the outlier run to load IDs for.
                                            If None, loads IDs for the current run_name.
                :param global_outliers: Flag indicating whether to include outliers from all runs.
                :param grouped_by_id: Flag indicating whether to load grouped outliers by patient ID.
                :return: A list of outlier groups extracted from the metadata file.

                :raises FileNotFoundError: If the metadata file does not exist and cannot be created from
                                            previous analysis results.
                """

        print("Loading IDs of outlier groups...")
        folder_path = self.pm.get_complex_ml_path(
            parameters, ["metadata_analysis", model_key], False, False, outlier_run_name
        )
        file_name = "Summary_outliers_by_groups.csv" if grouped_by_id else "Summary_outlier_epochs.csv"
        fullpath = folder_path / file_name
        try:
            outliers_df = pd.read_csv(fullpath)
        except FileNotFoundError:
            print(f"File '{fullpath}' not found. Trying to create from previous results...")
            try:
                from MachineLearning.Evaluation.meta_fold_analyzer import MetaFoldAnalyzer
                fold_analyzer = MetaFoldAnalyzer(self.pm, model_key, parameters, outlier_run_name)
                outliers_df = fold_analyzer.select_outlier_groups(save_res=True) if grouped_by_id \
                    else fold_analyzer.select_outlier_epochs(save_res=True)
            except (FileNotFoundError, ImportError, ModuleNotFoundError):
                print(f"No previous results found to create '{file_name}'. "
                      "No outlier IDs can be loaded.")
                return None

        # Ensure that global outliers include outliers from the given run by saving them before loading global outliers
        if global_outliers:
            # Create a saver instance
            try:
                from MachineLearning.IO.save_result import SaveResult
                saver = SaveResult(self.pm)

                outlier_type = "patient_id" if grouped_by_id else "epoch"
                saver.save_global_outliers(parameters, outliers_df, outlier_type)  # Add given outliers to global
                outliers_df = self.load_global_outliers(parameters, outlier_type)  # Get updated global outliers
            except (ImportError, ModuleNotFoundError):
                print("SaveResult could not be loaded. Skipping global outliers integration.")

        outlier_list = outliers_df["group"].values.tolist()
        f_print_val = "groups" if grouped_by_id else "epochs"
        print(f"Outlier {f_print_val} that will be removed from analysis: {outlier_list}")

        return outlier_list

    def load_run_data(self, hyperparameters: dict, run_name: str, model_key: str) -> dict:
        """
        Returns the metadata of a specific run based on the provided hyperparameters, run name, and model key.

        :param hyperparameters: A dictionary containing the hyperparameters of the run.
        :param run_name: The name of the specific run for which metadata is being retrieved.
        :param model_key: The key identifying the model.
        :return: The full file path to the metadata for the specified run.
        :raises FileNotFoundError: If no metadata file matching the given run name is found.
        """
        # Get all paths
        metadata_folderpath = self.pm.get_complex_ml_path(
            hyperparameters, ["run_metadata", model_key], False, False
        )
        metadata_fullpaths, _ = PathUtils.list_files_in_folder(metadata_folderpath, ".json", fullpaths=True)

        # Search for the path according to the provided run name
        matching_metadata_path = next((path for path in metadata_fullpaths if path.name == f"{run_name}.json"), None)
        # Error if not found
        if not matching_metadata_path:
            raise FileNotFoundError(
                f"No matching file for run_name='{run_name}' found in folder='{metadata_folderpath}'.")

        # Load and return metadata
        metadata = self.load_json(matching_metadata_path)
        return metadata

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
        :return: If combined is True, returns a Pandas DataFrame combining all individual result DataFrames.
                If combined is False, returns a dictionary where keys are file names and values
                are the corresponding result DataFrames.
        """
        results_list = self.pm.get_related_paths(hyperparamers, run_name, ["results", model_key])

        df_dict = {}
        for result_path in results_list:
            df_dict[result_path.stem] = pd.read_csv(result_path)

        if combined:
            df_combined = pd.concat(df_dict.values(), axis=0)
            return df_combined
        else:
            return df_dict

    def load_combined_features_df(self, parameters: dict, class_1: str, class_0: str)\
            -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        This method loads data from specified file paths for two given classes and
        returns their corresponding dataframes.

        :param parameters: A dict containing configuration parameters.
        :param class_1: The name or identifier of the first class whose data will be loaded.
            Valid options are: 'normal_an', 'faw' and 'awake'
        :param class_0: The name or identifier of the second class whose data will be loaded.
            Valid options are: 'normal_an', 'faw' and 'awake'
        :return: A tuple of pandas DataFrames corresponding to the data of `class_1` and `class_0`.
        """
        class_1_path = self.pm.resolve_episode_path(
            parameters, class_1, ["test_and_train_data", "feature_sets"], True, False
        )
        class_0_path = self.pm.resolve_episode_path(
            parameters, class_0, ["test_and_train_data", "feature_sets"], True, False
        )
        print(f"Loading {class_1} data from {class_1_path}\n "
              f"Loading {class_0} data from {class_0_path}")
        class_1_df = pd.read_csv(class_1_path).copy() # Copying to avoid pd messing with the file content
        class_0_df = pd.read_csv(class_0_path).copy()

        return class_1_df, class_0_df

    def load_global_outliers(self, parameters, outlier_type) -> pd.DataFrame:
        """
        Loads a dataframe of global outliers based on the specified outlier type and parameters.
        The method determines the target file name based on the `outlier_type` and retrieves
        the file location using the provided parameters. It returns the data from the corresponding CSV file.

        :param parameters: List of parameters used to determine the folder path for outliers files.
        :param outlier_type: Specifies the type of outliers to fetch, either "epoch" or "patient_id".
        :return: Pandas DataFrame containing the data of global outliers.
        :raises ValueError: If the provided outlier_type is not "epoch" or "patient_id".
        """
        if outlier_type not in ["epoch", "patient_id"]:
            raise ValueError("Invalid outlier type. Expected 'epoch' or 'patient_id'.")

        # Build a fullpath depending on the outlier type
        folder_path = self.pm.get_complex_ml_path(parameters, ["global_outliers"],False, False)
        name_discriminator = "epoch" if outlier_type == "epoch" else "patient"
        filename = f"global_{name_discriminator}_outliers.csv"
        fullpath = folder_path / filename

        # Load and return outlier df
        outliers_df = pd.read_csv(fullpath)
        return outliers_df

    def load_further_results(self, hyperparameters: dict, analysis_key: str, filename: str)\
            -> pd.DataFrame | dict:
        """
        Loads additional results specified by the analysis key and filename from
        the corresponding path derived using the given hyperparameters.

        This method determines the full path to the results based on the hyperparameters
        and loads the data in either a DataFrame or JSON structure, depending on the file extension.


        :param hyperparameters: Dictionary containing the hyperparameters which are used
            to generate the full path to the results.
        :param analysis_key: The Key, used to identify the specific analysis folder within the
            hyperparameters structure.
        :param filename: The name of the file to be loaded from the derived folder path.

        :returns: The loaded results as a pandas DataFrame if the extension is
            '.csv', or as a dictionary if the type is '.json'.

        :raises ValueError: If the `result_type` is not 'dataframe' or 'json'.
        """
        # Get extension of file to load properly
        ext = Path(filename).suffix
        if ext not in [".csv", ".json"]:
            raise ValueError(f"Invalid result_file extension. Expected '.csv' or '.json', got {ext}")

        folder_path = self.pm.get_complex_ml_path(
            hyperparameters, ["further_analysis", analysis_key], False, True
        )
        fullpath = folder_path / filename

        result = pd.read_csv(fullpath) if ext == ".csv" else self.load_json(fullpath)
        return result
    
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